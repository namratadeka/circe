from torch import nn
from abc import abstractmethod, ABC


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

        self._build_modules()

    @abstractmethod
    def _build_modules(self):
        pass

    @abstractmethod
    def forward(self):
        pass
