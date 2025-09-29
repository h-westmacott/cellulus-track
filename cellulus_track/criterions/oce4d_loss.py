import torch
import torch.nn as nn
import numpy as np


class OCE4DLoss(nn.Module):  # type: ignore
    def __init__(
        self,
        temperature: float,
        regularization_weight: float,
        density: float,
        num_spatial_dims: int,
        device: torch.device,
    ):
        """Class definition for loss.

        Parameters
        ----------

            temperature:
                Factor used to scale the gaussian function and control
                the rate of damping.

            regularization_weight:
                The weight of the L2 regularizer on the object-centric embeddings.

            density:
                Determines the fraction of patches to sample per crop,
                during training.

            num_spatial_dims:
                Should be equal to 2 for 2D and 3 for 3D.

            device:
                The device to train on.
                Set to 'cpu' to train without GPU.

        """
        super().__init__()
        self.temperature = temperature
        self.regularization_weight = regularization_weight
        self.density = density
        self.num_spatial_dims = num_spatial_dims
        self.device = device

    @staticmethod
    def distance_function(embedding_0, embedding_1):
        difference = embedding_0 - embedding_1
        return difference.norm(2, dim=-1)

    def non_linearity(self, distance):
        return 1 - (-distance.pow(2) / self.temperature).exp()

    def forward(self, anchor_embedding, reference_embedding, embeddings_velocity_anchor,embeddings_velocity_reference):
        distance = self.distance_function(
            anchor_embedding, reference_embedding.detach()
        )
        non_linear_distance = self.non_linearity(distance)
        oce_loss = non_linear_distance.sum()

        oce_4d_loss = torch.gradient(torch.gradient(non_linear_distance,dim=1)[0],dim=1)[0].sum()

        # test loss:
        anchor_embedding_rolled = torch.roll(anchor_embedding,1,dims=0)
        reference_embedding_rolled = torch.roll(reference_embedding,1,dims=0)

        # velocity = self.distance_function(
        #     anchor_embedding, reference_embedding_rolled.detach()
        # )
        # self_velocity = self.distance_function(
        #     anchor_embedding, anchor_embedding_rolled.detach()
        # )

        # velocity = anchor_embedding - reference_embedding_rolled
        
        anchor_velocity = anchor_embedding - anchor_embedding_rolled
        reference_velocity = reference_embedding - reference_embedding_rolled
    

        # non_linear_velocity = self.non_linearity(velocity)

        # non_linear_self_velocity = self.non_linearity(self_velocity)


        # velocity_oce_loss = 3000*(non_linear_velocity-non_linear_self_velocity).norm(2)
        velocity_oce_loss2 = (anchor_velocity-embeddings_velocity_anchor).norm(2)
        # + (reference_velocity-embeddings_velocity_reference).norm(2)
        velocity_oce_loss2 = velocity_oce_loss2*100

        regularization_loss = (
            self.regularization_weight * anchor_embedding.norm(2, dim=-1).sum()
        )
        loss = oce_loss + regularization_loss
        # + velocity_oce_loss2
        # loss = velocity_oce_loss2
        return loss, oce_loss, regularization_loss, velocity_oce_loss2
