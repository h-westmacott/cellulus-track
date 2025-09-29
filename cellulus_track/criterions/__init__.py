from cellulus_track.criterions.oce_loss import OCELoss
from cellulus_track.criterions.oce4d_loss import OCE4DLoss


def get_loss(
    temperature,
    regularizer_weight,
    density,
    num_spatial_dims,
    device,
    mode='OCE'
):
    if mode=='OCE':
        return OCELoss(
            temperature,
            regularizer_weight,
            density,
            num_spatial_dims,
            device,
        )
    elif mode=='OCE4D':
        return OCE4DLoss(
            temperature,
            regularizer_weight,
            density,
            num_spatial_dims,
            device,
        )
