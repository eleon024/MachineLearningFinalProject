import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class GlobalPoolQNet(nn.Module):
    def __init__(self, channels,height,width, n_actions):
        super().__init__()
        # store for the reshape logic
        self.C = channels
        self.H = height
        self.W = width
        self.conv1 = nn.Conv2d(self.C, 32, 8, 4, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 4, 1)
        self.conv3 = nn.Conv2d(64,128, 3, 2, 1)
        self.gap   = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(128, n_actions)

    def forward(self, x):
        if x.dim() == 2:
        # assume x is [batch, C*H*W], reshape here
            x = x.view(-1, self.C, self.H, self.W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.gap(x)               # → [batch,128,1,1]
        x = x.view(x.size(0), -1)     # → [batch,128]
        return self.fc(x)


    def save(self, file_name='GloabalQNetModel.pth'):
        """Save state_dict to ./model/<file_name>."""
        folder = './model'
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file_name)
        torch.save(self.state_dict(), path)

    def load(self, file_name='GloabalQNetModel.pth'):
        """Load state_dict from ./model/<file_name>."""
        path = os.path.join('MachineLearningFinalProject/snake-ai-pytorch/model/GloabalQNetModel.pth')
        self.load_state_dict(torch.load(path))


class DeepConvQNet(nn.Module):
    """
    Deep Q-Network with convolutional layers for processing the game state grid.
    Takes as input a state with given channels (e.g., 6 channels for snake body, 
    good food, poison, static obstacle, moving obstacle, snake head) and outputs 
    Q-values for each possible action.
    """
    def __init__(self, channels: int, height: int, width: int, n_actions: int):
        super().__init__()
        self.channels = channels
        self.height   = height
        self.width    = width

        # conv blocks with strong downsampling
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2)   # → 32×(H/4)×(W/4)
        self.conv2 = nn.Conv2d(32,       64, kernel_size=4, stride=4, padding=1)   # → 64×(H/16)×(W/16)
        self.conv3 = nn.Conv2d(64,       64, kernel_size=3, stride=2, padding=1)   # → 64×(H/32)×(W/32)

        # compute resulting spatial size
        conv_h = ((height + 2*2 - 8)//4 + 1 + 2*1 - 4)//4 + 1   # conv1→conv2 down to H/16
        conv_h = (conv_h + 2*1 - 3)//2 + 1                      # conv3 → H/32
        conv_w = ((width  + 2*2 - 8)//4 + 1 + 2*1 - 4)//4 + 1
        conv_w = (conv_w + 2*1 - 3)//2 + 1

        linear_input = conv_h * conv_w * 64  # should be on the order of 64×15×15 ≈ 14k
        self.fc1 = nn.Linear(linear_input, 512)
        self.fc2 = nn.Linear(512, n_actions)


    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, self.channels, self.height, self.width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save(self, file_name: str = "model.pth"):
        """Save the model parameters to the specified file (in a 'model' folder)."""
        folder = "./model"
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, file_name)
        torch.save(self.state_dict(), file_path)

class QTrainer:
    """
    Trainer class to perform a training step for the Q-network using Mean Squared Error loss 
    (can be adapted to Huber loss for stability if needed). It applies the Q-learning update:
    Q_new = reward + gamma * max(next_Q) for non-terminal states, and Q_new = reward for terminal states.
    """
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # Using MSE; consider nn.SmoothL1Loss (Huber) for stability if needed.
        self.update_count = 0
    def train_step(self, state, action, reward, next_state, done):
        # Convert arrays to PyTorch tensors
        state      = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action     = torch.tensor(action, dtype=torch.long)
        reward     = torch.tensor(reward, dtype=torch.float)

        # Ensure batch dimension is present
        if state.dim() == 1:
            # If we have a single sample, add a batch dimension
            state      = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            done       = (done, )

        # 1. Predicted Q values for the current state
        pred = self.model(state)  # shape: (batch_size, n_actions)
        # 2. Construct target Q values based on reward and best future Q (Bellman update)
        target = pred.clone().detach()  # start from current predictions
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # If not terminal, add discounted max next state Q-value
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0)))
            # The action is one-hot encoded; find index of the action taken and update its Q-value
            action_index = torch.argmax(action[idx]).item()  # get the index of the action that was 1
            target[idx][action_index] = Q_new

        # 3. Optimize the model: loss = MSE(target, pred)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        print(f"[Trainer] step {self.update_count:04d}: batch={state.shape[0]}  loss={loss.item():.4f}")
        self.update_count += 1
        loss.backward()
        # (Optional) Gradient clipping or monitoring could be added here for stability
        self.optimizer.step()

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
