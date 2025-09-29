import os

import numpy as np
import torch
import zarr
from tqdm import tqdm

from cellulus_track.criterions import get_loss
from cellulus_track.datasets import get_dataset
from cellulus_track.models import get_model
from cellulus_track.utils import get_logger

torch.backends.cudnn.benchmark = True


def erfnet_train(experiment_config):
    print(experiment_config)

    if not os.path.exists("models"):
        os.makedirs("models")

    train_config = experiment_config.train_config
    model_config = experiment_config.model_config

    # create train dataset
    train_dataset = get_dataset(
        dataset_config=train_config.train_data_config,
        crop_size=tuple(train_config.crop_size),
        elastic_deform=train_config.elastic_deform,
        control_point_spacing=train_config.control_point_spacing,
        control_point_jitter=train_config.control_point_jitter,
        density=train_config.density,
        kappa=train_config.kappa,
        normalization_factor=experiment_config.normalization_factor,
    )

    # create train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        drop_last=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )

    # set model
    model = get_model(
        in_channels=train_dataset.get_num_channels(),
        out_channels=train_dataset.get_num_spatial_dims()*2,
        num_fmaps=model_config.num_fmaps,
        fmap_inc_factor=model_config.fmap_inc_factor,
        features_in_last_layer=model_config.features_in_last_layer,
        downsampling_factors=[
            tuple(factor) for factor in model_config.downsampling_factors
        ],
        num_spatial_dims=train_dataset.get_num_spatial_dims()+1,
        num_heads = 1,
    )

    # from cellulus.models.erfnet import TrackERFNet
    # from erfnet_models.BasicUNet import BasicUNet

    # # model = TrackERFNet(n_classes=[1,1,2])
    # model = BasicUNet()
    # print(model)
    

    # set device
    device = torch.device(train_config.device)

    model = model.to(device)

    # initialize model weights
    if model_config.initialize:
        for _name, layer in model.named_modules():
            if isinstance(layer, torch.nn.modules.conv._ConvNd):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    # set loss
    criterion = get_loss(
        regularizer_weight=train_config.regularizer_weight,
        temperature=train_config.temperature,
        density=train_config.density,
        num_spatial_dims=train_dataset.get_num_spatial_dims(),
        device=device,
        mode='OCE'
    )

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.initial_learning_rate, weight_decay=0.01
    )

    # set logger
    logger = get_logger(keys=["loss",
                                "oce_loss",
                                "oce_4d_loss"
                                ], title="loss")

    # resume training
    start_iteration = 0
    lowest_loss = 1e6
    epoch_loss = 0
    num_iterations = 0
    if model_config.checkpoint is None:
        pass
    else:
        print(f"Resuming model from {model_config.checkpoint}")
        state = torch.load(model_config.checkpoint, map_location=device)
        start_iteration = state["iteration"] + 1
        lowest_loss = state["lowest_loss"]
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])
        logger.data = state["logger_data"]

    # call `train_iteration`
    for iteration, batch in tqdm(
        zip(
            range(start_iteration, train_config.max_iterations),
            train_dataloader,
        )
    ):
        loss, oce_loss, oce_4d_loss, prediction = train_iteration(iteration,
            batch, model=model, criterion=criterion, optimizer=optimizer, device=device, num_spatial_dims=train_dataset.get_num_spatial_dims()
        )
        # print(f"===> loss: {loss:.6f}, oce loss: {oce_loss:.6f}, oce_4d_loss: {oce_4d_loss:.6f}")
        logger.add(key="loss", value=loss)
        logger.add(key="oce_loss", value=oce_loss)
        logger.add(key="oce_4d_loss", value=oce_4d_loss)
        logger.write()
        logger.plot()

        # Check if lowest loss
        epoch_loss += loss
        num_iterations += 1
        if iteration % train_config.save_best_model_every == 0:
            is_lowest = epoch_loss / (num_iterations) < lowest_loss
            lowest_loss = min(epoch_loss / num_iterations, lowest_loss)
            if is_lowest:
                state = {
                    "iteration": iteration,
                    "lowest_loss": lowest_loss,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "logger_data": logger.data,
                }
                save_model(state, iteration, is_lowest)
            epoch_loss = 0
            num_iterations = 0

        # Save model at specific intervals
        if (
            iteration % train_config.save_model_every == 0
            or iteration == train_config.max_iterations - 1
        ):
            state = {
                "iteration": iteration,
                "lowest_loss": lowest_loss,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "logger_data": logger.data,
            }
            save_model(state, iteration)

        # Save snapshots at specific intervals
        if iteration % train_config.save_snapshot_every == 0:
            save_snapshot(
                batch,
                prediction,
                iteration,
            )


def train_iteration(itr,batch, model, criterion, optimizer, device,num_spatial_dims):
    if len(batch)==3:
        raw, anchor_coordinates, reference_coordinates = batch
    else:
        raw, anchor_coordinates, reference_coordinates, translation = batch
    # raw = raw[0,:]
    raw = raw[:,:,0]
    raw = torch.unsqueeze(raw,0)
    anchor_coordinates = anchor_coordinates.repeat(2,1,1)
    reference_coordinates = reference_coordinates.repeat(2,1,1)
    raw, anchor_coordinates, reference_coordinates = (
        raw.to(device),
        anchor_coordinates.to(device),
        reference_coordinates.to(device),
    )

    model.train()
    # offsets = model(torch.unsqueeze(raw.permute((1,0,2,3)),0))
    curr_frame = raw[:,:,0,:]
    prev_frame = raw[:,:,1,:]
    # out = model(curr_frame,prev_frame)
    out = model(curr_frame)
    offsets_t1 = out[0]
    offsets_t2 = out[1]
    # offsets_v = out[2]
    offsets_v = out[1]
    embeddings_anchor_t1 = model.select_and_add_coordinates(offsets_t1, anchor_coordinates)
    embeddings_reference_t1 = model.select_and_add_coordinates(
        offsets_t1, reference_coordinates
    )
    embeddings_anchor_t2 = model.select_and_add_coordinates(offsets_t2, anchor_coordinates)
    embeddings_reference_t2 = model.select_and_add_coordinates(
        offsets_t2, reference_coordinates
    )
    # embeddings_velocity_anchor = model.select_coordinates(offsets[:,-num_spatial_dims:,:], anchor_coordinates)
    # embeddings_velocity_reference = model.select_coordinates(
    #     offsets[:,-num_spatial_dims:,:], reference_coordinates
    # )

    loss_t1, oce_loss_t1, regularization_loss = criterion(
        embeddings_anchor_t1, embeddings_reference_t1
    )
    loss_t2, oce_loss_t2, regularization_loss = criterion(
        embeddings_anchor_t2, embeddings_reference_t2
    )
    loss = loss_t1 + loss_t2
    oce_loss = oce_loss_t1 + oce_loss_t2

    if not translation.all():
        velocity = offsets_t1[:,:num_spatial_dims,:] - offsets_t2[:,:num_spatial_dims,:]
    else:
        # translation = np.array(translation)
        translation = translation.to(device)[0]
        # print("using GT translation")
        velocity = torch.ones_like(offsets_t1,device=device)
        for i in range(0, num_spatial_dims):
            velocity[:,i,:] = velocity[:,i,:] * translation[i]

    velocity_loss = offsets_v - velocity

    # velocity_loss_cont = velocity_loss.norm(2) * (itr//2000)
    velocity_loss_cont = velocity_loss.norm(2) * 100 * int(itr>50000)
    loss += velocity_loss_cont

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), oce_loss.item(), velocity_loss_cont.item(), offsets_t1


def save_model(state, iteration, is_lowest=False):
    if is_lowest:
        file_name = os.path.join("models", "best_loss.pth")
        torch.save(state, file_name)
        print(f"Best model weights saved at iteration {iteration}")
    else:
        file_name = os.path.join("models", str(iteration).zfill(6) + ".pth")
        torch.save(state, file_name)
        print(f"Checkpoint saved at iteration {iteration}")


def save_snapshot(batch, prediction, iteration):
    raw, anchor_coordinates, reference_coordinates, translation = batch
    raw = raw[0,:]
    num_spatial_dims = len(raw.shape) - 2

    axis_names = ["s", "c"] + ["t", "z", "y", "x"][-num_spatial_dims:]
    prediction_offset = tuple(
        (a - b) / 2
        for a, b in zip(
            raw.shape[-num_spatial_dims:], prediction.shape[-num_spatial_dims:]
        )
    )
    f = zarr.open("snapshots.zarr", "a")
    f[f"{iteration}/raw"] = raw.detach().cpu().numpy()
    f[f"{iteration}/raw"].attrs["axis_names"] = axis_names
    f[f"{iteration}/raw"].attrs["resolution"] = [
        1,
    ] * num_spatial_dims

    # normalize the offsets by subtracting the mean offset per image
    prediction_cpu = prediction.detach().cpu().numpy()
    prediction_cpu_reshaped = np.reshape(
        prediction_cpu, (prediction_cpu.shape[0], prediction_cpu.shape[1], -1)
    )
    mean_prediction = np.mean(prediction_cpu_reshaped, 2)
    prediction_cpu -= mean_prediction[(...,) + (np.newaxis,) * (num_spatial_dims)]
    f[f"{iteration}/prediction"] = prediction_cpu
    f[f"{iteration}/prediction"].attrs["axis_names"] = axis_names
    f[f"{iteration}/prediction"].attrs["offset"] = prediction_offset
    f[f"{iteration}/prediction"].attrs["resolution"] = [
        1,
    ] * num_spatial_dims
