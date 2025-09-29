import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(
            ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True
        )

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True
        )

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1),
        )

        self.conv1x3_2 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated),
        )

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes, input_channels):
        super().__init__()
        self.initial_block = DownsamplerBlock(
            input_channels, 16
        )  # TODO input_channels = 1 (for gray-scale), 3 (for RGB)
        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes, n_init_features=128):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(n_init_features, 64))
        self.layers.append(non_bottleneck_1d(64, 0.0, 1))
        self.layers.append(non_bottleneck_1d(64, 0.0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0.0, 1))
        self.layers.append(non_bottleneck_1d(16, 0.0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True
        )

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# ERFNet


class Net(nn.Module):
    def __init__(
        self, num_classes, input_channels, encoder=None
    ):  # use encoder to pass pretrained encoder
        super().__init__()

        if encoder == None:
            self.encoder = Encoder(num_classes, input_channels)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            return self.decoder.forward(output)

class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, encoder=None):
        super().__init__()

        print("Creating branched erfnet with {} classes".format(num_classes))

        if encoder is None:
            self.encoder = Encoder(sum(num_classes), input_channels)

        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(Decoder(n))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print("Initialize last layer with size: ", output_conv.weight.size())
            print("*************************")
            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2 : 2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2 : 2 + n_sigma].fill_(1)

    def forward(self, input):

        output = self.encoder(input)

        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)

class TrackERFNet(BranchedERFNet):
    """
    Tracking network. Consists of a single, shared encoder and 3 distinct decoders.
    2 decoders are trained on segmentation, whereas 1 decoder is trained on tracking.
    """

    def __init__(self, n_classes=[4, 1, 2], input_channels=1, encoder=None):
        """
        Initialize tracking net.
        Args:
            n_classes (list): number of output channels for the 3 decoders
            (the first two inputs are the number of output channels for the segmentation decoders,
            the last one for the tracking decoder)
            input_channels (int): number of input channels
            encoder (nn.Module, optional): provide a custom encoder, otherwise an ERFNet encoder is used
        """
        super(TrackERFNet, self).__init__(
            n_classes,
            input_channels=input_channels,
            encoder=encoder,
        )
        self.decoders = nn.ModuleList()
        n_init_features = [128] * len(n_classes)
        n_init_features[-1] = 2 * n_init_features[-1]
        for n, n_feat in zip(n_classes, n_init_features):
            self.decoders.append(Decoder(n, n_init_features=n_feat))

    def forward(self, curr_frames, prev_frames):
        """
        Forward pairs of images (t, t+1). Two images in the same batch dimension position
        from the tensors curr_frames and prev_frames form an image pair from time points t and t-1.
        Args:
            curr_frames (torch.Tensor): tensor of images BxCxHxW
            prev_frames (torch.Tensor): tensor of images BxCxHxW

        Returns:

        """
        # dual input with shared weights, concat predictions -> then 2 decoders (->segmentation, flow prediction)
        images = torch.cat([curr_frames, prev_frames], dim=0)
        features_encoder = self.encoder(images)
        features_curr_frames = features_encoder[: curr_frames.shape[0]]
        features_prev_frames = features_encoder[curr_frames.shape[0] :]
        features_stacked = torch.cat(
            [features_curr_frames, features_prev_frames], dim=1
        )  # batchsize x 2 x h x w
        segm_prediction = torch.cat(
            [decoder.forward(features_encoder) for decoder in self.decoders[:-1]], 1
        )
        segm_prediction_curr = segm_prediction[: curr_frames.shape[0]]
        segm_prediction_prev = segm_prediction[curr_frames.shape[0] :]
        tracking_prediction = self.decoders[-1].forward(features_stacked)
        return segm_prediction_curr, segm_prediction_prev, tracking_prediction

    def init_output(self, n_sigma=1):
        # init last layers for tracking and segmentation offset similar
        with torch.no_grad():
            for decoder in (self.decoders[0], self.decoders[-1]):
                output_conv = decoder.output_conv
                print("Initialize last layer with size: ", output_conv.weight.size())
                print("*************************")
                output_conv.weight[:, 0:2, :, :].fill_(0)
                output_conv.bias[0:2].fill_(0)

                output_conv.weight[:, 2 : 2 + n_sigma, :, :].fill_(0)
                output_conv.bias[2 : 2 + n_sigma].fill_(1)
    
    @staticmethod
    def select_and_add_coordinates(outputs, coordinates):
        selections = []
        # outputs.shape = (b, c, h, w) or (b, c, d, h, w)
        # outputs = torch.squeeze(outputs).permute((1,0,2,3))
        # outputs = torch.squeeze(outputs)
        for output, coordinate in zip(outputs, coordinates):
            if output.ndim == 3:
                selection = output[:, coordinate[:, 1], coordinate[:, 0]]
            elif output.ndim == 4:
                selection = output[
                    :, coordinate[:, 2], coordinate[:, 1], coordinate[:, 0]
                ]
            selection = selection.transpose(1, 0)
            selection = selection[:,:2]
            selection += coordinate
            selections.append(selection)

        # selection.shape = (b, c, p) where p is the number of selected positions
        return torch.stack(selections, dim=0)


def calc_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)