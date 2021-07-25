import random
import os
import numpy as np
import torch
import mlflow


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# MlFlowClient のRun ID を引き回すためのラッパー
class MlflowWriter():
    def __init__(self, experiment_name, **kwargs):
        self.client = mlflow.tracking.MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id
        self.run_id = None

    def create_run_id(self, fold=None):
        if fold is not None:
            tags = {'fold': fold}
            run = self.client.create_run(self.experiment_id, tags=tags)
            self.run_id = run.info.run_id
        else:
            self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_params_from_config(self, config):
        for key, value in config.items():
            self._explore_recursive(key, value)

    def _explore_recursive(self, parent, element):
        if isinstance(element, dict):
            for key, value in element.items():
                self._explore_recursive(f'{parent}.{key}', value)
        elif isinstance(element, list):
            for idx, value in enumerate(element):
                self.client.log_param(self.run_id, f'{parent}.{idx}', value)
        else:
            self.client.log_param(self.run_id, parent, element)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, step):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path=local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)

if __name__ == "__main__":
    writer = MlflowWriter("test_experiment")
    writer.create_run_id(fold=0)
    config = {'A': {'a': 1}, 'B': '2'}
    writer.log_params_from_config(config)
    writer.log_metric('A', 3, step=0)
    writer.log_metric('A', 4, step=0)
    writer.set_terminated()