from model.regressor import RegressorBuilder
from model.classifier import ClassifierBuilder


class ModelFactory(object):
    """Factory class to build new model objects
    """

    def __init__(self):
        self._builders = dict()

    def register_builder(self, key, builder):
        """Registers a new model builder into the factory
        Args:
            key (str): string key of the model builder
            builder (any): Builder object
        """
        self._builders[key] = builder

    def create(self, key, **kwargs):
        """Instantiates a new builder object, once it's registered
        Args:
            key (str): string key of the model builder
            **kwargs: keyword arguments
        Returns:
            any: Returns an instance of a model object correspponding to the model builder
        Raises:
            ValueError: If model builder is not registered, raises an exception
        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


factory = ModelFactory()
# Register all model builders here.
factory.register_builder("regressor", RegressorBuilder())
factory.register_builder("classifier", ClassifierBuilder())
