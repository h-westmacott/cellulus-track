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


def train(experiment_config):
    print(experiment_config)
    

    experiment_name = experiment_config.experiment_name
    if not os.path.exists(experiment_name+"_models"):
        os.makedirs(experiment_name+"_models")

    train_config = experiment_config.train_config
    model_config = experiment_config.model_config

    use_pretrained_model = train_config.use_pretrained_model

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
    device = torch.device(train_config.device)

    # set model
    model = get_model(
        in_channels=train_dataset.get_num_channels(),
        out_channels=train_dataset.get_num_spatial_dims(),
        # out_channels=train_dataset.get_num_spatial_dims()*2,
        num_fmaps=model_config.num_fmaps,
        fmap_inc_factor=model_config.fmap_inc_factor,
        features_in_last_layer=model_config.features_in_last_layer,
        downsampling_factors=[
            tuple(factor) for factor in model_config.downsampling_factors
        ],
        num_spatial_dims=train_dataset.get_num_spatial_dims()+1,
        num_heads = 2,
    )
    model = model.to(device)
    print("model created")
    
    if use_pretrained_model:
        model_pretrained = get_model(
            in_channels=train_dataset.get_num_channels(),
            out_channels=train_dataset.get_num_spatial_dims(),
            # out_channels=train_dataset.get_num_spatial_dims()*2,
            num_fmaps=model_config.num_fmaps,
            fmap_inc_factor=model_config.fmap_inc_factor,
            features_in_last_layer=model_config.features_in_last_layer,
            downsampling_factors=[
                tuple(factor) for factor in model_config.downsampling_factors
            ],
            num_spatial_dims=train_dataset.get_num_spatial_dims()+1,
            num_heads = 2,
        )
        model_pretrained = model_pretrained.to(device)
        model_pretrained.load_state_dict(torch.load(train_config.pretrained_model_path)['model_state_dict'])
        # model_pretrained = model_pretrained.cuda()
        model_pretrained.eval()
        print("pretrained model loaded")
        
    else:
        model_pretrained = None

    # set device
    

    

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
    )

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.initial_learning_rate, weight_decay=0.01
    )

    # set logger
    logger = get_logger(keys=["loss",
                                "oce_loss",
                                "oce_4d_loss"
                                ], title=experiment_name+"_loss",
                                )

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

    train_velocity_head = train_config.train_velocity_head
    train_seg_head = train_config.train_seg_head

    # call `train_iteration`
    print('test print')
    for iteration, batch in tqdm(
        zip(
            range(start_iteration, train_config.max_iterations),
            train_dataloader,
        )
    ):
        loss, oce_loss, oce_4d_loss, prediction = train_iteration(iteration,
            batch, model=model, model_pretrained=model_pretrained, criterion=criterion, optimizer=optimizer, device=device, num_spatial_dims=train_dataset.get_num_spatial_dims(),
            train_seg_head=train_seg_head, train_velocity_head=train_velocity_head
        )
        # print("preparing to freeze")
        # if iteration > 10000:
        #    optimizer.lr = train_config.initial_learning_rate/1000
        #    print('checking for freeze')
        #    ct = 0
        #    for child in model.children():
        #         if ct in [0,1]:
        #             print('freezing child:', child)
        #             for param in child.parameters():
        #                    param.requires_grad = False
        #         ct+=1

        # loss, oce_loss, oce_4d_loss, prediction = train_iteration(iteration,
        #     batch, model=model, criterion=criterion, optimizer=optimizer, device=device, num_spatial_dims=train_dataset.get_num_spatial_dims(),
        #     train_seg_head=train_seg_head, train_velocity_head=train_velocity_head
        # )
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
                save_model(state, iteration, experiment_name, is_lowest)
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
            save_model(state, iteration, experiment_name)

        # Save snapshots at specific intervals
        if iteration % train_config.save_snapshot_every == 0:
            save_snapshot(
                batch,
                prediction,
                experiment_name,
                iteration,
            )


def train_iteration(itr,batch, model, criterion, optimizer, device,num_spatial_dims,model_pretrained= None, train_velocity_head = False, train_seg_head = True):
    if len(batch)==3:
        raw, anchor_coordinates, reference_coordinates = batch
    else:
        raw, anchor_coordinates, reference_coordinates, translation = batch
    # raw = raw[0,:]
    loss = torch.tensor(0,device=device,dtype=torch.float32)
    # raw = raw[:,:,0] # remove channel dimension?
    # raw = torch.unsqueeze(raw,1) # add "empty" channel dimension
    raw = torch.permute(raw,(0,2,1)+tuple(range(3,raw.ndim))) # flip channel and time dims
    anchor_coordinates = anchor_coordinates.repeat(2,1,1)
    reference_coordinates = reference_coordinates.repeat(2,1,1)
    raw, anchor_coordinates, reference_coordinates = (
        raw.to(device),
        anchor_coordinates.to(device),
        reference_coordinates.to(device),
    )

    model.train()
    # offsets = model(torch.unsqueeze(raw.permute((1,0,2,3)),0))
    # offsets = model(raw)
    offsets, velocities = model(raw)
    if model_pretrained is not None:
        offsets_p,velocities_p = model_pretrained(raw)
        offsets_t2, velocities_t2 = model_pretrained(torch.flip(raw,[2]))
    else:
        offsets_t2, velocities_t2 = model(torch.flip(raw,[2]))
    # offsets = offsets_p

    embeddings_anchor = model.select_and_add_coordinates(offsets[:,:num_spatial_dims,:], anchor_coordinates)
    embeddings_reference = model.select_and_add_coordinates(
        offsets[:,:num_spatial_dims,:], reference_coordinates
    )
    embeddings_velocity_anchor = model.select_coordinates(offsets[:,-num_spatial_dims:,:], anchor_coordinates)
    embeddings_velocity_reference = model.select_coordinates(
        offsets[:,-num_spatial_dims:,:], reference_coordinates
    )
    # seg_loss_cont, oce_loss, regularization_loss, oce_4d_loss = criterion(
    #     embeddings_anchor, embeddings_reference,embeddings_velocity_anchor,embeddings_velocity_reference
    # )
    seg_loss_cont, oce_loss, regularization_loss = criterion(
        embeddings_anchor, embeddings_reference
    )
    # loss, oce_loss, regularization_loss = criterion(
    #     embeddings_anchor, embeddings_reference
    # )
    
    velocity = torch.zeros_like(offsets,device=device)
    for j in range(translation.shape[0]):
        if not translation[j,:].all():
            # print('using real offsets')
            if model_pretrained is not None:
                this_velocity = offsets_p[j] - offsets_t2[j]
            else:
                this_velocity = offsets[j] - offsets_t2[j]
        else:
            # translation = np.array(translation)
            translation = translation.to(device)
            # print("using GT translation, gt shift is:",translation)
            this_velocity = torch.ones_like(offsets[j],device=device)
            for i in range(0, num_spatial_dims):
                # this_velocity[i,:] = velocity[j,i,:] * translation[j,i]
                this_velocity[i,:] = this_velocity[i,:] * translation[j,i] * -1
        velocity[j,:] = this_velocity

    # velocity_loss = offsets[:,num_spatial_dims:,:] - velocity
    velocity_loss = velocities - velocity
    # velocity_loss = 100
    # velocity loss contribution is 10000 for the first 1000 iterations (while the offset model trains)
    # then the actual velocity loss *100 after that
    # velocity_loss_cont = (velocity_loss.norm(2) * int(itr>5000) * 100) + (int(itr<=5000) * 10000)
    velocity_loss_cont = (velocity_loss.norm(2) * int(itr>0) * 100)
    # velocity_loss_cont = velocity_loss.norm(2) * 100
    # loss += velocity_loss_cont
    # loss = velocity_loss_cont
    if train_seg_head:
        loss = seg_loss_cont
    else:
        seg_loss_cont -= seg_loss_cont

    if train_velocity_head:
        loss += velocity_loss_cont
    else:
        velocity_loss_cont -= velocity_loss_cont

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    oce_loss = torch.tensor(0).to(device)
    return loss.item(), seg_loss_cont.item(), velocity_loss_cont.item(), offsets


def save_model(state, iteration, experiment_name, is_lowest=False):
    if is_lowest:
        file_name = os.path.join(experiment_name+"_models", "best_loss.pth")
        torch.save(state, file_name)
        print(f"Best model weights saved at iteration {iteration}")
    else:
        file_name = os.path.join(experiment_name+"_models", str(iteration).zfill(6) + ".pth")
        torch.save(state, file_name)
        print(f"Checkpoint saved at iteration {iteration}")


def save_snapshot(batch, prediction, experiment_name, iteration):
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
    f = zarr.open(experiment_name+"_snapshots.zarr", "a")
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
