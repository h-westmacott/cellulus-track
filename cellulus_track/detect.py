import numpy as np
import zarr
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from tqdm import tqdm

from cellulus_track.configs.inference_config import InferenceConfig
from cellulus_track.datasets.meta_data import DatasetMetaData
from cellulus_track.utils.greedy_cluster import Cluster2d, Cluster3d
from cellulus_track.utils.mean_shift import mean_shift_segmentation
from cellulus_track.utils.mean_shift_track import mean_shift_segmentation_and_tracking

import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_iou(mask1, mask2):
    """
    Calculates the Intersection over Union (IoU) for two boolean masks.
    """
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    if union == 0:
        return 0.0  # Avoid division by zero if both masks are empty or no overlap
    return intersection / union

def analyze_cluster_agreement(segmentation_t2, prediction_t2, verbose=False):
    """
    Analyzes cluster agreement between two segmentation masks using the Hungarian algorithm.

    Args:
        segmentation_t2 (np.ndarray): A 2D numpy array where each element is 0 (background)
                                     or a positive integer label for clustering method A.
        prediction_t2 (np.ndarray): A 2D numpy array for clustering method B, with the same format.
        verbose (bool): If True, prints detailed information during processing.

    Returns:
        list: A list of dictionaries, where each dictionary contains information
              about a matched pair or an unmatched cluster and the decision made.
    """
    results = []

    # Get unique labels, excluding 0 (background)
    seg_labels_all = np.unique(segmentation_t2)
    seg_labels = seg_labels_all[seg_labels_all > 0]

    pred_labels_all = np.unique(prediction_t2)
    pred_labels = pred_labels_all[pred_labels_all > 0]

    if verbose:
        print(f"Segmentation A labels: {seg_labels}")
        print(f"Segmentation B labels: {pred_labels}")

    if len(seg_labels) == 0 and len(pred_labels) == 0:
        if verbose: print("Both segmentations are empty (only background).")
        return results
    
    if len(seg_labels) == 0:
        for p_label in pred_labels:
            results.append({
                'seg_label': None,
                'pred_label': p_label,
                'decision': 'unmatched_prediction_t2_cluster',
                'iou': 0.0
            })
        if verbose: print("Segmentation A is empty. All Prediction B clusters are unmatched.")
        return results

    if len(pred_labels) == 0:
        for s_label in seg_labels:
            results.append({
                'seg_label': s_label,
                'pred_label': None,
                'decision': 'unmatched_segmentation_t2_cluster',
                'iou': 0.0
            })
        if verbose: print("Prediction B is empty. All Segmentation A clusters are unmatched.")
        return results

    # Construct cost matrix (1 - IoU)
    # Rows: seg_labels, Columns: pred_labels
    cost_matrix = np.ones((len(seg_labels), len(pred_labels)))

    for i, s_label in enumerate(seg_labels):
        mask_s = (segmentation_t2 == s_label)
        if np.sum(mask_s) == 0: continue # Should not happen if labels are from unique
        for j, p_label in enumerate(pred_labels):
            mask_p = (prediction_t2 == p_label)
            if np.sum(mask_p) == 0: continue # Should not happen

            iou = calculate_iou(mask_s, mask_p)
            cost_matrix[i, j] = 1.0 - iou
            if verbose:
                print(f"  IoU(SegA {s_label}, PredB {p_label}) = {iou:.4f}, Cost = {cost_matrix[i,j]:.4f}")


    # Apply Hungarian algorithm
    # row_ind are indices for seg_labels, col_ind are for pred_labels
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    if verbose:
        print(f"\nHungarian algorithm matches (seg_idx, pred_idx): {list(zip(row_ind, col_ind))}")
        print(f"Costs of matches: {cost_matrix[row_ind, col_ind]}")


    matched_seg_indices = set()
    matched_pred_indices = set()

    # Evaluate matched pairs
    for i in range(len(row_ind)):
        s_idx, p_idx = row_ind[i], col_ind[i]
        s_label = seg_labels[s_idx]
        p_label = pred_labels[p_idx]

        matched_seg_indices.add(s_idx)
        matched_pred_indices.add(p_idx)

        mask_s = (segmentation_t2 == s_label)
        mask_p = (prediction_t2 == p_label)

        N_s = np.sum(mask_s)
        N_p = np.sum(mask_p)
        N_intersection = np.sum(mask_s & mask_p)
        iou = N_intersection / (N_s + N_p - N_intersection) if (N_s + N_p - N_intersection) > 0 else 0

        if verbose:
            print(f"\nEvaluating match: SegA Label {s_label} (idx {s_idx}) with PredB Label {p_label} (idx {p_idx})")
            print(f"  N_s={N_s}, N_p={N_p}, N_intersection={N_intersection}, IoU={iou:.4f}")

        # This check ensures we don't process pairs that the Hungarian algorithm
        # might have been forced to match despite very low (or zero) IoU,
        # especially if the number of clusters differs significantly.
        # A cost of 1.0 means IoU of 0.0.
        # We only consider matches with *some* overlap as true candidates for rules.
        # If cost_matrix[s_idx, p_idx] is very close to 1 (e.g. > 0.99, meaning IoU < 0.01),
        # it's practically no overlap.
        if cost_matrix[s_idx, p_idx] >= 0.999: # Effectively IoU < 0.001
             if verbose: print(f"  Match SegA {s_label} - PredB {p_label} has very low IoU ({iou:.4f}). Treating as effectively unmatched for rules.")
             # These will be caught later as unmatched if not added to results here.
             # To ensure they are explicitly listed as a "poor match":
             results.append({
                'seg_label': s_label,
                'pred_label': p_label,
                'decision': 'poor_match_discarded',
                'iou': iou,
                'cost': cost_matrix[s_idx, p_idx]
             })
             # Remove from matched sets so they can be picked up as unmatched later if we want to be strict
             # However, the Hungarian algorithm *did* match them. So let's keep them matched but label the decision.
             # The unmatched logic below will then correctly skip them.
             continue


        N_s_only = N_s - N_intersection
        N_p_only = N_p - N_intersection
        
        decision_details = {'seg_label': s_label, 'pred_label': p_label, 'iou': iou}

        # Rule 1: Perfect Agreement
        if N_s_only == 0 and N_p_only == 0 and N_s > 0: # N_s > 0 to ensure it's a real cluster
            decision_details.update({'decision': 'perfect_match', 'accepted_mask_source': 'common'})
            if verbose: print(f"  Decision: Perfect Match. Accepted: Common.")
        else:
            # Rule 2: Accept clustering A (segmentation_t2)
            # "<5% of points in are clustered differently"
            # Assuming "points in" refers to points in the segmentation_t2 cluster (s_label)
            disagreement_s_percent = (N_s_only / N_s) if N_s > 0 else 1.0 # if N_s is 0, 100% disagreement
            decision_details['disagreement_s_percent'] = disagreement_s_percent

            if N_s > 0 and disagreement_s_percent < 0.05:
                decision_details.update({'decision': 'accept_segmentation_t2', 'accepted_mask_source': 'segmentation_t2'})
                if verbose: print(f"  Decision: Accept Segmentation A. Disagreement {disagreement_s_percent*100:.2f}%. Accepted: SegA {s_label}.")
            else:
                # Rule 3: Reject Both (~50% symmetric disagreement)
                N_union = N_s + N_p - N_intersection
                if N_union > 0:
                    symmetric_disagreement_ratio = (N_s_only + N_p_only) / N_union
                    decision_details['symmetric_disagreement_ratio'] = symmetric_disagreement_ratio
                    # Define "~50%" as, for example, between 30% and 70%
                    if 0.3 <= symmetric_disagreement_ratio <= 0.7:
                        decision_details.update({'decision': 'reject_both'})
                        if verbose: print(f"  Decision: Reject Both. Symmetric Disagreement {symmetric_disagreement_ratio*100:.2f}%.")
                    else:
                        decision_details.update({'decision': 'other_disagreement'})
                        if verbose: print(f"  Decision: Other Disagreement. SegA Disagr: {disagreement_s_percent*100:.2f}%, Symm Disagr: {symmetric_disagreement_ratio*100:.2f}%.")
                else: # N_union is 0, means N_s=0, N_p=0, N_intersection=0. Should not happen if N_s>0 earlier.
                    decision_details.update({'decision': 'empty_match_error'})
                    if verbose: print(f"  Decision: Error - Empty match despite N_s>0 check earlier.")
        results.append(decision_details)

    # Identify unmatched clusters
    if verbose: print("\nIdentifying unmatched clusters...")
    unmatched_s_indices = set(range(len(seg_labels))) - matched_seg_indices
    for s_idx in unmatched_s_indices:
        s_label = seg_labels[s_idx]
        # Check if this s_label was part of a 'poor_match_discarded'
        # This check is a bit complex because 'poor_match_discarded' still means it was "matched" by Hungarian.
        # The logic above for `cost_matrix[s_idx, p_idx] >= 0.999` already adds a result.
        # So, true unmatched are those not in `row_ind` at all.
        # The current `matched_seg_indices` correctly reflects all `s_idx` that were part of `row_ind`.
        results.append({
            'seg_label': s_label,
            'pred_label': None,
            'decision': 'unmatched_segmentation_t2_cluster',
            'iou': 0.0 # By definition, no match means no IoU with any pred_label
        })
        if verbose: print(f"  Unmatched SegA Label: {s_label}")


    unmatched_p_indices = set(range(len(pred_labels))) - matched_pred_indices
    for p_idx in unmatched_p_indices:
        p_label = pred_labels[p_idx]
        results.append({
            'seg_label': None,
            'pred_label': p_label,
            'decision': 'unmatched_prediction_t2_cluster',
            'iou': 0.0
        })
        if verbose: print(f"  Unmatched PredB Label: {p_label}")
        
    return results

def analyze_cluster_agreement_v2(segmentation_t2, prediction_t2, 
                                 merge_detection_iou_threshold=0.1, verbose=False):
    """
    Analyzes cluster agreement, handling cases where prediction_t2 might merge
    multiple segmentation_t2 clusters (e.g., cell division).

    Args:
        segmentation_t2 (np.ndarray): Labels from method A.
        prediction_t2 (np.ndarray): Labels from method B.
        merge_detection_iou_threshold (float): IoU threshold to consider a prediction_t2
                                             cluster as overlapping a segmentation_t2 cluster
                                             for merge detection.
        verbose (bool): If True, prints detailed information.

    Returns:
        list: List of dictionaries with analysis results.
    """
    results = []

    seg_labels_all = np.unique(segmentation_t2)
    seg_labels = seg_labels_all[seg_labels_all > 0]

    pred_labels_all = np.unique(prediction_t2)
    pred_labels = pred_labels_all[pred_labels_all > 0]

    if verbose:
        print(f"Segmentation A (segmentation_t2) labels: {seg_labels}")
        print(f"Segmentation B (prediction_t2) labels: {pred_labels}")
        print(f"Merge detection IoU threshold: {merge_detection_iou_threshold}")

    if len(seg_labels) == 0 and len(pred_labels) == 0:
        if verbose: print("Both segmentations are empty.")
        return results
    
    # Handle cases where one segmentation is empty
    if len(seg_labels) == 0:
        for p_label in pred_labels:
            results.append({
                'seg_label': None, 'pred_label': p_label,
                'decision': 'unmatched_prediction_t2_cluster (seg_empty)', 'iou': 0.0
            })
        if verbose: print("Segmentation A is empty. All Prediction B clusters are unmatched.")
        return results

    if len(pred_labels) == 0:
        for s_label in seg_labels:
            results.append({
                'seg_label': s_label, 'pred_label': None,
                'decision': 'unmatched_segmentation_t2_cluster (pred_empty)', 'iou': 0.0
            })
        if verbose: print("Prediction B is empty. All Segmentation A clusters are unmatched.")
        return results

    # --- Pre-analysis: Identify potential merges by prediction_t2 (Method B) ---
    # p_merger_info: dict mapping p_label -> list of s_labels it significantly overlaps with
    p_merger_info = {}
    if verbose: print("\n--- Identifying potential merges by Prediction B ---")
    for p_label_candidate in pred_labels:
        mask_p_candidate = (prediction_t2 == p_label_candidate)
        if np.sum(mask_p_candidate) == 0: continue

        overlapped_s_labels_for_this_p = []
        for s_label_candidate in seg_labels:
            mask_s_candidate = (segmentation_t2 == s_label_candidate)
            if np.sum(mask_s_candidate) == 0: continue
            
            iou_for_merge_check = calculate_iou(mask_s_candidate, mask_p_candidate)
            if iou_for_merge_check > merge_detection_iou_threshold:
                overlapped_s_labels_for_this_p.append(s_label_candidate)
        
        if len(overlapped_s_labels_for_this_p) > 1:
            p_merger_info[p_label_candidate] = overlapped_s_labels_for_this_p
            if verbose:
                print(f"  Pred cluster {p_label_candidate} potentially merges Seg clusters: {overlapped_s_labels_for_this_p}")
    if verbose: print("--- End of merge identification ---")

    # --- Cost Matrix for Hungarian Algorithm ---
    cost_matrix = np.ones((len(seg_labels), len(pred_labels)))
    if verbose: print("\n--- Calculating Cost Matrix (1 - IoU) ---")
    for i, s_label in enumerate(seg_labels):
        mask_s = (segmentation_t2 == s_label)
        if np.sum(mask_s) == 0: continue
        for j, p_label in enumerate(pred_labels):
            mask_p = (prediction_t2 == p_label)
            if np.sum(mask_p) == 0: continue
            iou = calculate_iou(mask_s, mask_p)
            cost_matrix[i, j] = 1.0 - iou
            # if verbose: print(f"  IoU(Seg {s_label}, Pred {p_label}) = {iou:.4f}, Cost = {cost_matrix[i,j]:.4f}")
    if verbose: print("--- End of Cost Matrix Calculation ---")

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    if verbose:
        print(f"\nHungarian algorithm matches (seg_idx, pred_idx): {list(zip(row_ind, col_ind))}")
        print(f"Costs of matches: {cost_matrix[row_ind, col_ind] if len(row_ind)>0 else 'N/A'}")

    matched_seg_indices = set()
    matched_pred_indices = set()
    
    # --- Evaluate Matched Pairs from Hungarian Algorithm ---
    if verbose: print("\n--- Evaluating Matched Pairs ---")
    for i in range(len(row_ind)):
        s_idx, p_idx = row_ind[i], col_ind[i]
        s_label = seg_labels[s_idx]
        p_label = pred_labels[p_idx]

        matched_seg_indices.add(s_idx)
        matched_pred_indices.add(p_idx)

        mask_s = (segmentation_t2 == s_label)
        mask_p = (prediction_t2 == p_label)

        N_s = np.sum(mask_s)
        N_p = np.sum(mask_p)
        N_intersection = np.sum(mask_s & mask_p)
        current_match_iou = N_intersection / (N_s + N_p - N_intersection) if (N_s + N_p - N_intersection) > 0 else 0

        if verbose:
            print(f"\nEvaluating match: SegA {s_label} (idx {s_idx}) with PredB {p_label} (idx {p_idx})")
            print(f"  N_s={N_s}, N_p={N_p}, N_intersection={N_intersection}, Current Match IoU={current_match_iou:.4f}")

        decision_details = {'seg_label': s_label, 'pred_label': p_label, 'iou': current_match_iou}

        # Check if s_label is part of any identified merge scenario (cell division)
        s_is_part_of_a_split = False
        merging_p_details_for_s = None
        for p_candidate_merger, s_list_merged_by_p in p_merger_info.items():
            if s_label in s_list_merged_by_p:
                s_is_part_of_a_split = True
                merging_p_details_for_s = {
                    'merging_pred_label': p_candidate_merger,
                    'all_s_labels_in_split': s_list_merged_by_p
                }
                break
        
        if s_is_part_of_a_split:
            decision_details['decision'] = 'reject_segmentation_potential_split'
            reason = (f'Seg cluster {s_label} is part of a group ({merging_p_details_for_s["all_s_labels_in_split"]}) '
                      f'identified as merged by Pred cluster {merging_p_details_for_s["merging_pred_label"]}.')
            decision_details['reason'] = reason
            decision_details['context_original_hungarian_match'] = \
                f'Matched to Pred cluster {p_label} (IoU {current_match_iou:.4f}) by Hungarian.'
            if p_label == merging_p_details_for_s["merging_pred_label"]:
                decision_details['pred_cluster_role_in_this_match'] = 'merger_of_this_and_other_seg_clusters'
            if verbose: print(f"  Decision: {decision_details['decision']}. {reason}")
        
        # If s_label is not part of a split, apply standard rules
        else:
            if cost_matrix[s_idx, p_idx] >= 0.999: # Effectively IoU < 0.001 for this specific match
                decision_details['decision'] = 'poor_match_discarded'
                decision_details['cost'] = cost_matrix[s_idx, p_idx]
                if verbose: print(f"  Decision: Poor Match (IoU {current_match_iou:.4f}). Discarded from rule evaluation.")
            else:
                N_s_only = N_s - N_intersection
                N_p_only = N_p - N_intersection
                
                if N_s_only == 0 and N_p_only == 0 and N_s > 0:
                    decision_details.update({'decision': 'perfect_match', 'accepted_mask_source': 'common'})
                    if verbose: print(f"  Decision: Perfect Match.")
                else:
                    disagreement_s_percent = (N_s_only / N_s) if N_s > 0 else 1.0
                    decision_details['disagreement_s_percent'] = disagreement_s_percent

                    if N_s > 0 and disagreement_s_percent < 0.05:
                        decision_details.update({'decision': 'accept_segmentation_t2', 'accepted_mask_source': 'segmentation_t2'})
                        if verbose: print(f"  Decision: Accept SegA. Disagreement {disagreement_s_percent*100:.2f}%.")
                    else:
                        N_union = N_s + N_p - N_intersection
                        if N_union > 0:
                            symmetric_disagreement_ratio = (N_s_only + N_p_only) / N_union
                            decision_details['symmetric_disagreement_ratio'] = symmetric_disagreement_ratio
                            if 0.45 <= symmetric_disagreement_ratio <= 0.55: # ~50%
                                decision_details.update({'decision': 'reject_both'})
                                if verbose: print(f"  Decision: Reject Both. Symmetric Disagreement {symmetric_disagreement_ratio*100:.2f}%.")
                            else:
                                decision_details.update({'decision': 'other_disagreement'})
                                if verbose: print(f"  Decision: Other Disagreement. SegA Disagr: {disagreement_s_percent*100:.2f}%, Symm Disagr: {symmetric_disagreement_ratio*100:.2f}%.")
                        else:
                            decision_details.update({'decision': 'empty_match_error'})
                            if verbose: print(f"  Decision: Error - Empty match.")
            
            # Add note if the matched p_label is a merger of *other* s_labels
            if p_label in p_merger_info and (not s_is_part_of_a_split or p_label != merging_p_details_for_s.get('merging_pred_label')):
                 # The 'not s_is_part_of_a_split' ensures this note is for s_labels not already handled by the split logic
                 # OR if s_label was part of a split by a *different* p_merger, but this p_label is *also* a merger
                decision_details['pred_cluster_note'] = f'Matched Pred cluster {p_label} also merges other Seg clusters: {p_merger_info[p_label]}'


        results.append(decision_details)

    # --- Handle Unmatched Clusters ---
    if verbose: print("\n--- Identifying Unmatched Clusters ---")
    unmatched_s_indices = set(range(len(seg_labels))) - matched_seg_indices
    for s_idx in unmatched_s_indices:
        s_label = seg_labels[s_idx]
        decision_details = {'seg_label': s_label, 'pred_label': None, 'iou': 0.0}

        s_is_part_of_a_split_unmatched = False
        merging_p_details_for_unmatched_s = None
        for p_candidate_merger, s_list_merged_by_p in p_merger_info.items():
            if s_label in s_list_merged_by_p:
                s_is_part_of_a_split_unmatched = True
                merging_p_details_for_unmatched_s = {
                    'merging_pred_label': p_candidate_merger,
                    'all_s_labels_in_split': s_list_merged_by_p
                }
                break
        
        if s_is_part_of_a_split_unmatched:
            decision_details['decision'] = 'reject_segmentation_potential_split_unmatched'
            reason = (f'Unmatched Seg cluster {s_label} is part of a group ({merging_p_details_for_unmatched_s["all_s_labels_in_split"]}) '
                      f'identified as merged by Pred cluster {merging_p_details_for_unmatched_s["merging_pred_label"]}.')
            decision_details['reason'] = reason
            if verbose: print(f"  Unmatched SegA Label {s_label}: Decision: {decision_details['decision']}. {reason}")
        else:
            decision_details['decision'] = 'unmatched_segmentation_t2_cluster'
            if verbose: print(f"  Unmatched SegA Label {s_label}: Decision: {decision_details['decision']}.")
        results.append(decision_details)

    unmatched_p_indices = set(range(len(pred_labels))) - matched_pred_indices
    for p_idx in unmatched_p_indices:
        p_label = pred_labels[p_idx]
        decision_details = {'seg_label': None, 'pred_label': p_label, 'iou': 0.0}
        if p_label in p_merger_info:
            decision_details['decision'] = 'unmatched_prediction_merger_cluster'
            reason = f'Unmatched Pred cluster {p_label} was identified as merging Seg clusters: {p_merger_info[p_label]}'
            decision_details['reason'] = reason
            if verbose: print(f"  Unmatched PredB Label {p_label}: Decision: {decision_details['decision']}. {reason}")
        else:
            decision_details['decision'] = 'unmatched_prediction_t2_cluster'
            if verbose: print(f"  Unmatched PredB Label {p_label}: Decision: {decision_details['decision']}.")
        results.append(decision_details)
        
    return results


def detect(inference_config: InferenceConfig) -> None:
    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData.from_dataset_config(dataset_config)

    f = zarr.open(inference_config.detection_dataset_config.container_path)
    ds = f[inference_config.detection_dataset_config.secondary_dataset_name]

    # can't be arsed to fix this right now
    ds = ds[:-1,:,:677,1:]
    dataset_meta_data.num_samples = ds.shape[0]
    dataset_meta_data.spatial_array = ds.shape[2:]


    # prepare the zarr dataset to write to
    f_detection = zarr.open(inference_config.detection_dataset_config.container_path)
    ds_detection = f_detection.create_dataset(
        inference_config.detection_dataset_config.dataset_name,
        shape=(
            dataset_meta_data.num_samples,
            inference_config.num_bandwidths,
            *dataset_meta_data.spatial_array,
        ),
        dtype=np.uint16,
        overwrite=True,
    )
    ds_detection[:] = np.zeros(ds_detection.shape).copy()

    ds_detection.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"][
        -dataset_meta_data.num_spatial_dims :
    ]
    ds_detection.attrs["resolution"] = (1,) * dataset_meta_data.num_spatial_dims
    ds_detection.attrs["offset"] = (0,) * dataset_meta_data.num_spatial_dims

    # prepare the binary segmentation zarr dataset to write to
    ds_binary_segmentation = f_detection.create_dataset(
        "binary-segmentation",
        shape=(
            dataset_meta_data.num_samples,
            1,
            *dataset_meta_data.spatial_array,
        ),
        dtype=np.uint16,
        overwrite=True,
    )

    ds_binary_segmentation.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"][
        -dataset_meta_data.num_spatial_dims :
    ]
    ds_binary_segmentation.attrs["resolution"] = (
        1,
    ) * dataset_meta_data.num_spatial_dims
    ds_binary_segmentation.attrs["offset"] = (0,) * dataset_meta_data.num_spatial_dims

    # prepare the object centered embeddings zarr dataset to write to
    ds_object_centered_embeddings = f_detection.create_dataset(
        "centered-embeddings",
        shape=(
            dataset_meta_data.num_samples,
            (dataset_meta_data.num_spatial_dims * 2) + 1,
            *dataset_meta_data.spatial_array,
        ),
        dtype=float,
        overwrite=True,
    )

    ds_object_centered_embeddings.attrs["axis_names"] = ["s", "c"] + [
        "t",
        "z",
        "y",
        "x",
    ][-dataset_meta_data.num_spatial_dims :]
    ds_object_centered_embeddings.attrs["resolution"] = (
        1,
    ) * dataset_meta_data.num_spatial_dims
    ds_object_centered_embeddings.attrs["offset"] = (
        0,
    ) * dataset_meta_data.num_spatial_dims

    # prepare the predicted segmentation zarr dataset to write to
    ds_predicted_segmentation = f_detection.create_dataset(
        "predicted-segmentation",
        shape=(
            dataset_meta_data.num_samples,
            1,
            *dataset_meta_data.spatial_array,
        ),
        dtype=np.uint16,
        overwrite=True,
    )

    ds_predicted_segmentation.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"][
        -dataset_meta_data.num_spatial_dims :
    ]
    ds_predicted_segmentation.attrs["resolution"] = (
        1,
    ) * dataset_meta_data.num_spatial_dims
    ds_predicted_segmentation.attrs["offset"] = (0,) * dataset_meta_data.num_spatial_dims

    ds_tracked_labels = f_detection.create_dataset(
        "tracked-segmentation",
        shape=(
            dataset_meta_data.num_samples,
            1,
            *dataset_meta_data.spatial_array,
        ),
        dtype=np.uint16,
        overwrite=True,
    )

    ds_tracked_labels.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"][
        -dataset_meta_data.num_spatial_dims :
    ]
    ds_tracked_labels.attrs["resolution"] = (
        1,
    ) * dataset_meta_data.num_spatial_dims
    ds_tracked_labels.attrs["offset"] = (0,) * dataset_meta_data.num_spatial_dims

    def center_embeddings(embedding, binary_mask):

        embeddings_mean = embedding[
            np.newaxis, : dataset_meta_data.num_spatial_dims, ...
        ].copy()

        embeddings_centered = embedding.copy()
        embeddings_mean_masked = (
            binary_mask[np.newaxis, np.newaxis, ...] * embeddings_mean
        )

        if (embeddings_centered.shape[0]+1)//2 == 3:
            c_x = embeddings_mean_masked[0, 0]
            c_y = embeddings_mean_masked[0, 1]
            c_x = c_x[c_x != 0].mean()
            c_y = c_y[c_y != 0].mean()
            embeddings_centered[0] -= c_x
            embeddings_centered[1] -= c_y
        # elif embeddings_centered.shape[0] == 4:
        elif (embeddings_centered.shape[0]+1)//2 == 4:
            c_x = embeddings_mean_masked[0, 0]
            c_y = embeddings_mean_masked[0, 1]
            c_z = embeddings_mean_masked[0, 2]
            c_x = c_x[c_x != 0].mean()
            c_y = c_y[c_y != 0].mean()
            c_z = c_z[c_z != 0].mean()
            embeddings_centered[0] -= c_x
            embeddings_centered[1] -= c_y
            embeddings_centered[2] -= c_z
        ds_object_centered_embeddings[sample] = embeddings_centered

        embeddings_centered_mean = embeddings_centered[
            np.newaxis, : dataset_meta_data.num_spatial_dims
        ]
        embeddings_centered_std = embeddings_centered[-1]

        return embeddings_centered_mean, embeddings_centered_std


    for sample in tqdm(range(dataset_meta_data.num_samples-1)):
        embeddings = ds[sample]
        embeddings_std = embeddings[-1, ...]
        embeddings_mean = embeddings[
            np.newaxis, : dataset_meta_data.num_spatial_dims, ...
        ].copy()

        embeddings_next = ds[sample + 1]
        embeddings_next_std = embeddings_next[-1, ...]
        embeddings_next_mean = embeddings_next[
            np.newaxis, : dataset_meta_data.num_spatial_dims, ...
        ].copy()

        if inference_config.threshold is None:
            threshold = threshold_otsu(embeddings_std)
            threshold_next = threshold_otsu(embeddings_next_std)
        else:
            threshold = inference_config.threshold
            threshold_next = threshold

        print(f"For sample {sample}, binary threshold {threshold} was used.")
        # # binary_mask = embeddings_std < threshold
        # binary_mask = embeddings_std > threshold
        binary_mask = np.logical_or(embeddings_std > threshold, embeddings_std == 0.0)
        # binary_mask_next = embeddings_next_std > threshold_next
        binary_mask_next = np.logical_or(embeddings_next_std > threshold_next, embeddings_std == 0.0)
        ds_binary_segmentation[sample, 0, ...] = binary_mask

        # find mean of embeddings
        embeddings_centered_mean, embeddings_centered_std = center_embeddings(
            embeddings, binary_mask
        )

        embeddings_next_centered_mean, embeddings_next_centered_std = center_embeddings(
            embeddings_next, binary_mask_next
        )

        if inference_config.clustering == "meanshift":
            for bandwidth_factor in range(inference_config.num_bandwidths):
                if inference_config.use_seeds:
                    offset_magnitude = np.linalg.norm(embeddings_centered_mean, axis=0)
                    offset_magnitude_smooth = gaussian_filter(offset_magnitude, sigma=2)
                    coordinates = peak_local_max(-offset_magnitude_smooth)
                    seeds = np.flip(coordinates, 1)
                    segmentation = mean_shift_segmentation(
                        embeddings_centered_mean,
                        embeddings_centered_std,
                        bandwidth=inference_config.bandwidth / (2**bandwidth_factor),
                        min_size=inference_config.min_size,
                        reduction_probability=inference_config.reduction_probability,
                        threshold=threshold,
                        seeds=seeds,
                    )
                    # embeddings_centered_mean = embeddings_centered[
                    #     np.newaxis, : dataset_meta_data.num_spatial_dims, ...
                    # ].copy()
                else:
                    segmentation, segmentation_prediction = mean_shift_segmentation_and_tracking(
                        embeddings_mean,
                        embeddings_next_mean,
                        embeddings_std,
                        embeddings_next_std,
                        bandwidth=inference_config.bandwidth / (2**bandwidth_factor),
                        min_size=inference_config.min_size,
                        reduction_probability=inference_config.reduction_probability,
                        threshold=threshold,
                        threshold_next = threshold_next,
                        seeds=None,
                    )
                    # Note that the line below is needed
                    # because the embeddings_mean is modified
                    # by mean_shift_segmentation
                    embeddings_mean = embeddings[
                        np.newaxis, : dataset_meta_data.num_spatial_dims, ...
                    ].copy()
                ds_detection[sample, bandwidth_factor, ...] = segmentation
                ds_predicted_segmentation[sample+1, 0, ...] = segmentation_prediction
                

        elif inference_config.clustering == "greedy":
            if dataset_meta_data.num_spatial_dims == 3:
                cluster3d = Cluster3d(
                    width=embeddings.shape[-1],
                    height=embeddings.shape[-2],
                    depth=embeddings.shape[-3],
                    fg_mask=binary_mask,
                    device=inference_config.device,
                )
                for bandwidth_factor in range(inference_config.num_bandwidths):
                    segmentation = cluster3d.cluster(
                        prediction=embeddings,
                        bandwidth=inference_config.bandwidth / (2**bandwidth_factor),
                        min_object_size=inference_config.min_size,
                    )
                    ds_detection[sample, bandwidth_factor, ...] = segmentation
            elif dataset_meta_data.num_spatial_dims == 2:
                cluster2d = Cluster2d(
                    width=embeddings.shape[-1],
                    height=embeddings.shape[-2],
                    fg_mask=binary_mask,
                    device=inference_config.device,
                )
                for bandwidth_factor in range(inference_config.num_bandwidths):
                    segmentation = cluster2d.cluster(
                        prediction=embeddings,
                        bandwidth=inference_config.bandwidth / (2**bandwidth_factor),
                        min_object_size=inference_config.min_size,
                    )

                    ds_detection[sample, bandwidth_factor, ...] = segmentation

    results = []
    for sample in tqdm(range(dataset_meta_data.num_samples-1)):
        # attempt to track
        import matplotlib.pyplot as plt
        print('Tracking')

        segmentation_t1 = ds_detection[sample]

        segmentation_t2 = ds_detection[sample + 1]
        prediction_t2 = ds_predicted_segmentation[sample + 1]
        print(' ')
        results_ex1 = analyze_cluster_agreement_v2(segmentation_t2, prediction_t2, verbose=False)
        # print("\nResults for Sample " + str(sample) + ":")
        results.append(results_ex1)
        # for res in results_ex1:
        #     print(res)
        # seg_2_pred = {}
        # for label in np.unique(prediction_t2):
        #     if label == 0:
        #         continue
        #     if len(np.unique(segmentation_t2[np.where(prediction_t2 == label)])) == 1:
        #         # check that the labels exactly match
        #         val = int(np.unique(segmentation_t2[np.where(prediction_t2 == label)]))
        #         if len(np.unique(prediction_t2[np.where(segmentation_t2 == val)])) == 1:
        #             # label pair is an exact match
        #             # we need a copy of segmentation_t2 that we can write the new labes to 
        #             # (as predicted by prediction_t2) without polluting the values in 
        #             # the current segmentation_t2
        #             # No we don't! We just need a mapping of the label predicted by the model:
        #             seg_2_pred[val] = label
        #     else:
        #         print('something went wrong!')

        #         plt.hist(segmentation_t2[np.where(prediction_t2 == label)])
        #         plt.title(str(sample))
        #         plt.show()
        #         # make the new segmentation label be selected as a vote by the predicted values?
        #         # there are cases where two cells in t1 have been clustered as two cells in t2:
        #         # these can be handled as a majority vote
        #         # There are cases where one cell in t1 has been clustered as two cells in t2:
        #         # these can be treated as a cell division
        #         # all other cases are over- or under-segmentations


        continue

    # track labels
    # first time point cannot be tracked, so we copy it in to use as a starting point.
    ds_tracked_labels[:] = np.zeros_like(ds_detection)
    ds_tracked_labels[0,:] = ds_detection[0]
    # mapping
    for sample in tqdm(range(1, dataset_meta_data.num_samples)):
        this_tracking = np.zeros_like(ds_detection[sample])
        segmentation = ds_detection[sample]
        prediction = ds_predicted_segmentation[sample]
        tracking_results = results[sample-1]
        for label in tracking_results:
            # label = label[0]
            if label['seg_label'] is None:
                # this is a prediction that was not matched to a segmentation
                continue
            if label['pred_label'] is None:
                # ds_tracked_labels[sample][ds_detection[sample] == label['seg_label']] = np.amax(ds_tracked_labels)+1
                this_tracking[ds_detection[sample] == label['seg_label']] = np.amax(ds_tracked_labels)+1
            else:
                # ds_tracked_labels[ds_detection[sample] == label['seg_label']] = label['pred_label']
                # ds_tracked_labels[sample][ds_detection[sample] == label['seg_label']] = int(np.mean(ds_tracked_labels[sample-1][ds_detection[sample-1] == label['pred_label']]))
                this_tracking[ds_detection[sample] == label['seg_label']] = int(np.mean(ds_tracked_labels[sample-1][ds_detection[sample-1] == label['pred_label']]))
        ds_tracked_labels[sample, :, ...] = this_tracking
    print('done!')
