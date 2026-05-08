import flwr as fl
import torch
from src.model import CNN

class Client(fl.client.NumPyClient):
    def __init__(self):
        self.model = CNN()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        return self.get_parameters(config), 100, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.5, 100, {}

fl.client.start_numpy_client(server_address="localhost:8080", client=Client())
