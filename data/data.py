from torch.utils.data import DataLoader

from data.dsprites import DspritesBuilder
from data.dsprites_nonlinear import DspritesNonLinearBuilder
from data.yale_extended import YaleBExtendedBuilder


class DataFactory(object):
    """
    Factory class to build new dataset objects
    """
    def __init__(self):
        self._builders = dict()

    def register_builder(self, key, builder):
        """Registers a new dataset builder into the factory
        Args:
            key (str): string key of the dataset builder
            builder (any): Builder object
        """
        self._builders[key] = builder

    def create(self, key, **kwargs):
        """Instantiates a new builder object, once it's registered
        Args:
            key (str): string key of the dataset builder
            **kwargs: keyword arguments
        Returns:
            any: Returns an instance of a dataset object correspponding to the dataset builder
        Raises:
            ValueError: If dataset builder is not registered, raises an exception
        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


def create_dataloaders(data_cfg, modes):
    print('Loading data:')
    loaders = dict()
    for mode in modes:
        ds_args = {'path': eval("data_cfg.{}".format(mode))}
        ds_args.update(data_cfg.__dict__)
        ood = 'ood' in mode
        ds_args.update({'ood': ood})
        dataset = factory.create(data_cfg.data_key, **ds_args)

        shuffle = mode == 'train'
        drop_last = True
        print(f'{mode} : {ds_args["path"]}')
        loaders[mode] = DataLoader(
            dataset = dataset,
            batch_size = data_cfg.batch_size,
            shuffle = shuffle,
            drop_last = drop_last, 
            pin_memory = True,
            num_workers = data_cfg.num_workers
        )
    print('Data loading complete.')
    return loaders


factory = DataFactory()
# Resigster all dataset builders here.

factory.register_builder("dsprites", DspritesBuilder())
factory.register_builder("dsprites-nonlinear", DspritesNonLinearBuilder())
factory.register_builder("yale-b", YaleBExtendedBuilder())
