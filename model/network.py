import torch
import torch.nn as nn


class Network(nn.Module):
    """Network Class
    This class constructs a network given an architecture.
    """

    def __init__(self, arch: dict):
        """Network Class Constructor

        Args:
            arch (dict): Architecture defined in config file for the individual module
        """
        super(Network, self).__init__()
        self._create_network(arch)

    def _create_network(self, arch) -> None:
        """General function to create a sequential module of the model using an architecture defined in the config file
        Supports Linear, Conv blocks, Dropout, Activation, Batchnorm etc.
        The dict name and keys should be written in the same format as used by Pytorch
        Ex: for nn.Linear block use Linear as dict name and in_features and out_features as keys of the dict

        Args:
            arch (dict): Architecture defined in config file for the individual module
        """
        self.blocks = nn.Sequential()
        for index, layer in enumerate(arch):

            for function, args in layer.items():
                if isinstance(args, dict):
                    self.blocks.add_module(
                        name="{}_{}".format(function.lower(), index),
                        module=eval("nn.{}(**{})".format(function, args)),
                    )
                elif function == "activation" and args is not None:
                    self.blocks.add_module(
                        name="activation_{}".format(index),
                        module=eval("nn.{}()".format(args)),
                    )
                elif function == "dropout":
                    self.blocks.add_module(
                        name="dropout_{}".format(index),
                        module=eval("nn.Dropout({})".format(args)),
                    )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """testing function for the sequential network

        Args:
            x (torch.tensor): random tensor to check the input and output dimension from the network

        Returns:
            torch.tensor: output of the sequential network built
        """
        return self.blocks(x)
