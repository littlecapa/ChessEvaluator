import torch
import torch.nn as nn

FILEPATH = "/Users/littlecapa/GIT/python/ChessEvaluator/model"

class ChessEvaluator(nn.Module):
    def __init__(self):
        super(ChessEvaluator, self).__init__()

        self.fc1 = nn.Linear(12, 64)  # Input layer to hidden layer 1
        self.fc2 = nn.Linear(64, 128)  # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(128, 256)  # Hidden layer 2 to hidden layer 3
        self.fc4 = nn.Linear(256, 256)  # Hidden layer 3 to hidden layer 4
        self.fc5 = nn.Linear(256, 128)  # Hidden layer 4 to hidden layer 5
        self.fc6 = nn.Linear(128, 1)  # Hidden layer 5 to output layer

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.zeros_(self.fc5.bias)
        nn.init.xavier_uniform_(self.fc6.weight)
        nn.init.zeros_(self.fc6.bias)
        torch.manual_seed(123)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

    def eval_position(self, bit_board_input_tensor):
        output = self.forward(bit_board_input_tensor)
        return output.item()

def save_model(model, filepath = FILEPATH):
    torch.save(model.state_dict(), filepath)

def load_model(filepath = FILEPATH):
    model = ChessEvaluator()
    model.load_state_dict(torch.load(filepath))
    return model