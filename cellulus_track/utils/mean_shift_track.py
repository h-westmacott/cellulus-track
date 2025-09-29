import numpy as np
import torch
from sklearn.cluster import MeanShift


def mean_shift_segmentation_and_tracking(
    embedding_mean,
    embedding_next_mean,
    embedding_std,
    embedding_next_std,
    bandwidth,
    min_size,
    reduction_probability,
    threshold,
    threshold_next,
    seeds,
):
    embedding_mean = torch.from_numpy(embedding_mean)
    embedding_next_mean = torch.from_numpy(embedding_next_mean)
    if embedding_mean.ndim == 4:
        embedding_mean[:, 1] += torch.arange(embedding_mean.shape[2])[
            None, :, None
        ]  # += dy
        embedding_mean[:, 0] += torch.arange(embedding_mean.shape[3])[
            None, None, :
        ]  # += dx
        # and embedding next:
        embedding_next_mean[:, 1] += torch.arange(embedding_next_mean.shape[2])[
            None, :, None
        ]  # += dy
        embedding_next_mean[:, 0] += torch.arange(embedding_next_mean.shape[3])[
            None, None, :
        ]  # += dx
    elif embedding_mean.ndim == 5:
        embedding_mean[:, 2] += torch.arange(embedding_mean.shape[2])[
            None, :, None, None
        ]
        embedding_mean[:, 1] += torch.arange(embedding_mean.shape[3])[
            None, None, :, None
        ]
        embedding_mean[:, 0] += torch.arange(embedding_mean.shape[4])[
            None, None, None, :
        ]
        # and embedding next:
        embedding_next_mean[:, 2] += torch.arange(embedding_next_mean.shape[2])[
            None, :, None, None
        ]
        embedding_next_mean[:, 1] += torch.arange(embedding_next_mean.shape[3])[
            None, None, :, None
        ]
        embedding_next_mean[:, 0] += torch.arange(embedding_next_mean.shape[4])[
            None, None, None, :
        ]

    # mask = embedding_std < threshold
    mask = np.logical_and(embedding_std < threshold, embedding_std > 0.0)
    # mask_next = embedding_next_std < threshold_next
    mask_next = np.logical_and(embedding_next_std < threshold_next, embedding_next_std > 0.0)

    mask = mask[None]
    mask_next = mask_next[None]
    # segmentation = segment_with_meanshift(
    #     embedding_mean,
    #     bandwidth,
    #     mask=mask,
    #     reduction_probability=reduction_probability,
    #     cluster_all=False,
    #     seeds=seeds,
    # )[0]

    anchor_mean_shift_track = AnchorMeanshift(
        bandwidth,
        reduction_probability=reduction_probability,
        cluster_all=False,
        seeds=seeds,
    )
    segmentation, segmentation_prediction = anchor_mean_shift_track(embedding_mean, embedding_next_mean, mask=mask, mask_next=mask_next)
    segmentation = segmentation[0] + 1
    segmentation_prediction = segmentation_prediction[0] + 1

    return segmentation, segmentation_prediction


# def segment_with_meanshift(
#     embedding, bandwidth, mask, reduction_probability, cluster_all, seeds
# ):
#     anchor_mean_shift = AnchorMeanshift(
#         bandwidth,
#         reduction_probability=reduction_probability,
#         cluster_all=cluster_all,
#         seeds=seeds,
#     )
#     return anchor_mean_shift(embedding, mask=mask) + 1


class AnchorMeanshift:
    def __init__(self, bandwidth, reduction_probability, cluster_all, seeds):
        self.mean_shift = MeanShift(
            bandwidth=bandwidth, cluster_all=cluster_all, seeds=seeds
        )
        self.reduction_probability = reduction_probability

    def fit_mean_shift(self, X):
        if self.reduction_probability < 1.0:
            X_reduced = X[np.random.rand(len(X)) < self.reduction_probability]
            mean_shift_segmentation = self.mean_shift.fit(X_reduced)
        else:
            mean_shift_segmentation = self.mean_shift.fit(X)

        mean_shift_segmentation = self.mean_shift.predict(X)

        return mean_shift_segmentation
    
    def compute_mean_shift(self, X):

        mean_shift_segmentation = self.mean_shift.predict(X)

        return mean_shift_segmentation

    def reshape_embedding(self, embedding, mask=None):
        if embedding.ndim == 3:
            c, h, w = embedding.shape
            if mask is not None:
                assert len(mask.shape) == 2
                if mask.sum() == 0:
                    return -1 * np.ones(mask.shape, dtype=np.int32)
                reshaped_embedding = embedding.permute(1, 2, 0)[mask].view(-1, c)
            else:
                reshaped_embedding = embedding.permute(1, 2, 0).view(w * h, c)
        elif embedding.ndim == 4:
            c, d, h, w = embedding.shape
            if mask is not None:
                
                assert len(mask.shape) == 3
                if mask.sum() == 0:
                    return -1 * np.ones(mask.shape, dtype=np.int32)
                reshaped_embedding = embedding.permute(1, 2, 3, 0)[mask].view(-1, c)
            else:
                reshaped_embedding = embedding.permute(1, 2, 3, 0).view(d * h * w, c)

        return reshaped_embedding

    def spatialize_mean_shift_segmentation(self, mean_shift_segmentation, mask=None, embedding_shape=None):
        if mask is not None:
            mean_shift_segmentation_spatial = -1 * np.ones(mask.shape, dtype=np.int32)
            mean_shift_segmentation_spatial[mask] = mean_shift_segmentation
            return mean_shift_segmentation_spatial
        else:
            if embedding_shape is not None:
                if len(embedding_shape) == 3:
                    return mean_shift_segmentation.reshape(embedding_shape[1], embedding_shape[2])
                elif len(embedding_shape) == 4:
                    return mean_shift_segmentation.reshape(embedding_shape[1], embedding_shape[2], embedding_shape[3])
            return mean_shift_segmentation

    def compute_masked_ms(self, embedding, mask=None):
        if embedding.ndim == 3:
            c, h, w = embedding.shape
            if mask is not None:
                assert len(mask.shape) == 2
                if mask.sum() == 0:
                    return -1 * np.ones(mask.shape, dtype=np.int32)
                reshaped_embedding = embedding.permute(1, 2, 0)[mask].view(-1, c)
            else:
                reshaped_embedding = embedding.permute(1, 2, 0).view(w * h, c)
        elif embedding.ndim == 4:
            c, d, h, w = embedding.shape
            if mask is not None:
                
                assert len(mask.shape) == 3
                if mask.sum() == 0:
                    return -1 * np.ones(mask.shape, dtype=np.int32)
                reshaped_embedding = embedding.permute(1, 2, 3, 0)[mask].view(-1, c)
            else:
                reshaped_embedding = embedding.permute(1, 2, 3, 0).view(d * h * w, c)

        reshaped_embedding = reshaped_embedding.contiguous().numpy()

        mean_shift_segmentation = self.fit_mean_shift(reshaped_embedding)
        if mask is not None:
            mean_shift_segmentation_spatial = -1 * np.ones(mask.shape, dtype=np.int32)
            mean_shift_segmentation_spatial[mask] = mean_shift_segmentation
            mean_shift_segmentation = mean_shift_segmentation_spatial
        else:
            if embedding.ndim == 2:
                mean_shift_segmentation = mean_shift_segmentation.reshape(h, w)
            elif embedding.ndim == 3:
                mean_shift_segmentation = mean_shift_segmentation.reshape(d, h, w)
        return mean_shift_segmentation


    def __call__(self, embedding, embedding_next, mask=None, mask_next=None):
        segmentation = []
        segmentation_prediction = []
        for j in range(len(embedding)):
            mask_slice = mask[j] if mask is not None else None
            mask_next_slice = mask_next[j] if mask_next is not None else None
            # mean_shift_segmentation = self.compute_masked_ms(
            #     embedding[j], mask=mask_slice
            # )
            #   we can now replace the line above with:
            reshaped_embedding = self.reshape_embedding(embedding[j], mask_slice)
            mean_shift_segmentation = self.fit_mean_shift(reshaped_embedding)
            reshaped_embedding_next = self.reshape_embedding(embedding_next[j], mask_next_slice)
            mean_shift_segmentation_next = self.compute_mean_shift(reshaped_embedding_next)
            #   then  I need to do some complex clever spatial stuff with this clustering
            mean_shift_segmentation = self.spatialize_mean_shift_segmentation(mean_shift_segmentation, mask=mask_slice, embedding_shape=embedding[j].shape)
            mean_shift_segmentation_next = self.spatialize_mean_shift_segmentation(mean_shift_segmentation_next, mask=mask_next_slice, embedding_shape=embedding_next[j].shape)
            
            segmentation.append(mean_shift_segmentation)
            segmentation_prediction.append(mean_shift_segmentation_next)


        return np.stack(segmentation), np.stack(segmentation_prediction)
    
