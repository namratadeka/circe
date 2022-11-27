import torch
import torchvision

from model.base import BaseModel
from model.network import Network


class Regressor(BaseModel):
    def __init__(self, model_cfg):
        self.cfg = model_cfg

        super(Regressor, self).__init__()

    def _build_modules(self):
        try:
            featurizer = eval("torchvision.models.{}(pretrained=True)".format(self.cfg.network['featurizer']))
            self.featurizer = torch.nn.Sequential(*(list(featurizer.children())[:-1]))
        except:
            self.featurizer = Network(self.cfg.network['featurizer'])

        self.fc1 = Network(self.cfg.network['fc1'])
        self.fc2 = Network(self.cfg.network['fc2'])
        self.target = Network(self.cfg.network['target'])

    def forward(self, x):
        # compute f(x):
        enc_ft = self.featurizer(x)
        enc_ft = torch.flatten(enc_ft, start_dim=1)
        fc1 = self.fc1(enc_ft)
        fc2 = self.fc2(fc1)

        # predict y from f(x):
        y_ = self.target(fc2)

        return [enc_ft, fc1, fc2, y_], y_

class RegressorBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, model_cfg, **_ignored):
        """Callback function
        Args:
            model_cfg (ModelConfig): Model Config object
            **_ignored: ignore extra arguments
        Returns:
            Regressor: Instantiated Regressor network object
        """
        self._instance = Regressor(model_cfg=model_cfg)
        return self._instance
