import gunpowder as gp
import torch
import zarr

from cellulus_track.configs.inference_config import InferenceConfig
from cellulus_track.datasets.meta_data import DatasetMetaData


def erfnet_predict(
    model: torch.nn.Module,
    inference_config: InferenceConfig,
    normalization_factor: float,
) -> None:
    # get the dataset_config data out of inference_config
    dataset_config = inference_config.dataset_config
    dataset_meta_data = DatasetMetaData.from_dataset_config(dataset_config)
    num_spatial_dims = dataset_meta_data.num_spatial_dims+1

    # set device
    device = torch.device(inference_config.device)

    # model.set_infer(
    #     p_salt_pepper=inference_config.p_salt_pepper,
    #     num_infer_iterations=inference_config.num_infer_iterations,
    #     device=device,
    # )

    # prediction crop size is the size of the scanned tiles to be provided to the model
    input_shape = gp.Coordinate(
        (1, dataset_meta_data.num_channels, 2, *inference_config.crop_size)
    )

    output_shape = gp.Coordinate(
        model(
            torch.zeros(
                (1, dataset_meta_data.num_channels, *inference_config.crop_size),
                dtype=torch.float32,
            ).to(device),
            # torch.zeros(
            #     (1, dataset_meta_data.num_channels, *inference_config.crop_size),
            #     dtype=torch.float32,
            # ).to(device)
        )[0].shape
    )

    # treat all dimensions as spatial, with a voxel size of 1
    input_voxel_size = (1,) * (dataset_meta_data.num_dims + 1)
    output_voxel_size = (1,) * (dataset_meta_data.num_dims)
    raw_spec = gp.ArraySpec(voxel_size=input_voxel_size, interpolatable=True)
    pred_spec = gp.ArraySpec(voxel_size=output_voxel_size, interpolatable=True)

    input_size = gp.Coordinate(input_shape) * gp.Coordinate(input_voxel_size)
    output_size = gp.Coordinate(output_shape) * gp.Coordinate(output_voxel_size)
    # diff_size = input_size - output_size
    diff_size = (input_size[0]-output_size[0],
                 input_size[1]-output_size[1],
                 0,
                 input_size[3]-output_size[2],
                 input_size[4]-output_size[3])

    if num_spatial_dims-1 == 2:
        context = (0, 0, diff_size[2] // 2, diff_size[3] // 2)
    elif num_spatial_dims-1 == 3:
        context = (
            0,
            0,
            diff_size[2] // 2,
            diff_size[3] // 2,
            diff_size[4] // 2,
        )  # type: ignore

    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICT")

    scan_request = gp.BatchRequest()
    if num_spatial_dims-1 == 2:
        scan_request[raw] = gp.Roi(
            (0, 0, -diff_size[2] // 2, -diff_size[3] // 2),
            (2, dataset_meta_data.num_channels, input_size[-2], input_size[-1]),
        )
        scan_request[prediction] = gp.Roi(
            (0, 0, 0, 0),
            (1, (2*(num_spatial_dims-1)) + 1, output_size[2], output_size[3]),
        )
    elif num_spatial_dims-1 == 3:
        scan_request[raw] = gp.Roi(
            (0, 0, -diff_size[2] // 2, -diff_size[3] // 2, -diff_size[4] // 2),
            (
                2,
                dataset_meta_data.num_channels,
                input_size[2],
                input_size[3],
                input_size[4],
            ),
        )
        scan_request[prediction] = gp.Roi(
            (0, 0, 0, 0),
            (
                1,
                (2*(num_spatial_dims-1)) + 1,
                output_size[2],
                output_size[3],
                # output_size[4],
            ),
        )

    predict = gp.torch.Predict(
        model,
        inputs={"raw": raw},
        outputs={0: prediction},
        array_specs={prediction: raw_spec},
    )

    predict_2 = my_predict(
        model,
        inputs={"raw": raw},
        outputs={0: prediction},
        array_specs={prediction: pred_spec},
    )

    # prepare the zarr dataset to write to
    f = zarr.open(inference_config.prediction_dataset_config.container_path)
    ds = f.create_dataset(
        inference_config.prediction_dataset_config.dataset_name,
        shape=(
            dataset_meta_data.num_samples,
            ((num_spatial_dims-1)*2) + 1,
            *dataset_meta_data.spatial_array,
        ),
        dtype=float,
    )
    dataset_config.container_path = str(dataset_config.container_path)
    inference_config.prediction_dataset_config.container_path = str(inference_config.prediction_dataset_config.container_path)

    pipeline = (
        gp.ZarrSource(
            dataset_config.container_path,
            {raw: dataset_config.dataset_name},
            {raw: gp.ArraySpec(voxel_size=output_voxel_size, interpolatable=True)},
        )
        + gp.Normalize(raw, factor=normalization_factor)
        + gp.Pad(raw, context, mode="reflect")
        # + gp.Unsqueeze([raw],axis=0)
        # + Permute2(raw,(1,0,2,3))
        + predict_2
        + gp.ZarrWrite(
            dataset_names={
                prediction: inference_config.prediction_dataset_config.dataset_name
            },
            output_filename=inference_config.prediction_dataset_config.container_path,
        )
        + gp.Scan(scan_request)
    )

    request = gp.BatchRequest()
    # request to pipeline for ROI of whole image/volume
    with gp.build(pipeline):
        pipeline.request_batch(request)

    ds.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"][
        -num_spatial_dims :
    ]

    ds.attrs["resolution"] = (1,) * num_spatial_dims
    ds.attrs["offset"] = (0,) * num_spatial_dims

class Permute(gp.BatchFilter):

  def __init__(self, array, permutation):
    self.array = array
    self.permutation = permutation

  def process(self, batch, request):
    import numpy as np
    data = batch[self.array].data
    batch[self.array].data = np.transpose(data,self.permutation)
    return batch
  
class Permute2(gp.BatchFilter):
    """Unsqueeze a batch at a given axis

    Args:
        arrays (List[ArrayKey]): ArrayKeys to unsqueeze.
        axis: Position where the new axis is placed, defaults to 0.
    """

    def __init__(self, array, permutation):
        self.array = array
        self.permutation = permutation

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        if self.array in request:
            deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        from gunpowder.batch import Batch
        import numpy as np
        outputs = Batch()
        if self.array in batch:
            if not batch[self.array].spec.nonspatial:
                spatial_dims = request[self.array].roi.dims
                # if self.axis > batch[self.array].data.ndim - spatial_dims:
                #     raise ValueError(
                #         (
                #             f"Unsqueeze.axis={self.axis} not permitted. "
                #             "Unsqueeze only supported for "
                #             "non-spatial dimensions of Array."
                #         )
                #     )

            outputs[self.array] = batch[self.array]
            # outputs[self.array].data = np.transpose(batch[self.array].data, self.permutation)
            outputs[self.array].data = np.transpose(batch[self.array].data, np.concatenate([np.array([0,]),np.array(self.permutation)+1]))
            ends = request[self.array].roi.end
            offsets = request[self.array].roi.offset
            outputs[self.array].spec.roi = gp.Roi(
                (offsets[self.permutation[0]],
                offsets[self.permutation[1]],
                offsets[self.permutation[2]],
                offsets[self.permutation[3]],)
                , (ends[self.permutation[0]],
                ends[self.permutation[1]],
                ends[self.permutation[2]],
                ends[self.permutation[3]],)
            )
            # outputs[self.array].data = np.expand_dims(batch[self.array].data, 0)
        return outputs

class my_predict(gp.torch.Predict):
   def predict(self, batch, request):
        inputs = self.get_inputs(batch)
        inputs['raw'] = torch.unsqueeze(torch.permute(inputs['raw'],(1,0,2,3)),0)
        curr_frame = inputs['raw'][:,:,0]
        prev_frame = inputs['raw'][:,:,1]
        with torch.no_grad():
            # out = self.model.forward(curr_frame,prev_frame)
            out = self.model.forward(curr_frame)
        # outputs = self.get_outputs(torch.concatenate([out[0],out[2],torch.zeros((1,1,224,224),device=out[0].device)],axis=1), request)
        outputs = self.get_outputs(torch.concatenate([out[0],out[0],torch.zeros((1,1,124,124),device=out[0].device)],axis=1), request)
        self.update_batch(batch, request, outputs)