import time
from PIL import ImageGrab
import pyautogui as pag
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

learning_rate = 0.001

class Policy(nn.Module):
    def __init__(self, input_n, output_n):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(input_n, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, output_n)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        rewards, action_probs_selected = zip(*self.data)

        self.optimizer.zero_grad()

        return_lst = torch.tensor(rewards, dtype=torch.float32)
        action_probs_selected = torch.stack(action_probs_selected)

        log_pi_returns = torch.multiply(
            torch.log(action_probs_selected + 1e-6), return_lst)
        policy_objective = torch.sum(log_pi_returns)
        loss = torch.multiply(policy_objective, -1.0)

        if torch.isfinite(loss):
            loss.backward()
            self.optimizer.step()

        self.data = []

        return loss

prob_act = []
pi = Policy(6400, 5)

try:
    while True:
        time.sleep(0.2)

        pag.keyUp('up')
        pag.keyUp('down')
        pag.keyUp('left')
        pag.keyUp('right')

        screen = ImageGrab.grab()
        screen = screen.convert('L')

        screen = screen.resize((80, 80))

        screen = np.array(screen)
        screen = screen / 255.0
        screen = screen.reshape(1, 6400)

        prob = pi.forward(screen)
        m = Categorical(prob)
        action = m.sample()

        prob_act.append(prob[0][action.item()])

        if len(prob_act) > 100:
            prob_act.pop(0)
    
        if action.item() == 0:
            pag.keyDown('up')
            print('up')
        elif action.item() == 1:
            pag.keyDown('down')
            print('down')
        elif action.item() == 2:
            pag.keyDown('left')
            print('left')
        elif action.item() == 3:
            pag.keyDown('right')
            print('right')
        elif action.item() == 4:
            pag.press('space', presses=10)
            print('space')

        pi.put_data((1.0, prob[0][action.item()]))

except KeyboardInterrupt:
    for i in range(len(prob_act)):
        pi.put_data((-1.0, prob_act[i]))
    pi.train_net()

    torch.save(pi.state_dict(), './model.pt')