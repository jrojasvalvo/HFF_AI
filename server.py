from math import gamma
from this import d
import UdpComms as U
import time
import pyautogui as p
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

#DeepQNetwork Class borrowed from https://www.youtube.com/watch?v=wc-FxNENg9U
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

#Agent class borrowed from https://www.youtube.com/watch?v=wc-FxNENg9U
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        #print("LEARNING")
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon> self.eps_min else self.eps_min

def gameInput(binary_action):
    #print("Binary Action: " + binary_action)
    for c in range(len(binary_action)):
        if c == 1:
            if binary_action[c] == "1":
                p.keyDown('w')
            else:
                p.keyUp('w')
        elif c == 2:
            if binary_action[c] == "1":
                p.keyDown('s') 
            else:
                p.keyUp('s')
        elif c == 3:
            if binary_action[c] == "1":
                p.keyDown('a')
            else:
                p.keyUp('a')
        elif c == 4:
            if binary_action[c] == "1":
                p.keyDown('d')
            else:
                p.keyUp('d')
        elif c == 5:
            if binary_action[c] == "1":
                p.keyDown('space')
            else:
                p.keyUp('space')
        elif c == 0:
            if binary_action[c] == "1":
                p.mouseDown()
                p.mouseDown(button='right')
            else:
                p.mouseUp()
                p.mouseUp(button='right')

def toBinary(num):
    if num >= 1: 
        return toBinary(num // 2) + str(num % 2) 
    return ""

def calc_reward(player_x, player_z, distance, from_start):
    left_flag = False
    forward_flag = False
    if player_z <= -16:
        left_flag = True
    if player_x >= 12.5:
        forward_flag = True
    if not left_flag or forward_flag:
        return (40.88 - distance) + ((from_start - 40.88) // (40.88 * 2))
    else:
        return from_start

if __name__ == '__main__':
    
    # Create UDP socket to use for sending (and receiving)
    sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
    input_dims = [8]
    agent = Agent(gamma=0.999, epsilon=0.5, batch_size=64, n_actions=64, eps_end=0.01, input_dims=input_dims, lr=0.009)
    load = input("Load previous training data?")
    if load == "Y" or load == "y":
        with open("state_memory.txt", 'r') as f:
            text = f.read()
            state_mem = text.split(" ")
            state_mem_ = []
            for i in state_mem:
                state_mem_.append(np.float32(i))
            state_mem = np.asarray(state_mem_)
        with open("new_state_memory.txt", 'r') as f:
            text = f.read()
            new_state_mem = text.split(" ")
            new_state_mem_ = []
            for i in new_state_mem:
                new_state_mem_.append(np.float32(i))
            new_state_mem = np.asarray(new_state_mem_)
        with open("action_memory.txt", 'r') as f:
            text = f.read()
            action_mem = text.split(" ")
            action_mem_ = []
            for i in action_mem:
                action_mem_.append(np.int32(i))
            action_mem = np.asarray(action_mem_)
        with open("reward_memory.txt", 'r') as f:
            text = f.read()
            reward_mem = text.split(" ")
            reward_mem_ = []
            for i in reward_mem:
                reward_mem_.append(np.float32(i))
            reward_mem = np.asarray(reward_mem_)
        with open("terminal_memory.txt", 'r') as f:
            text = f.read()
            terminal_mem = text.split(" ")
            terminal_mem_ = []
            for i in terminal_mem:
                terminal_mem_.append(np.bool(int(float(i))))
            terminal_mem = np.asarray(terminal_mem_)
        with open("counter.txt", 'r') as f:
            text = f.read()
            mem_cntr = int(text)

        agent.state_memory = np.reshape(state_mem, (agent.mem_size, *input_dims))
        agent.new_state_memory = np.reshape(new_state_mem, (agent.mem_size, *input_dims))
        agent.action_memory = action_mem
        agent.reward_memory = reward_mem
        agent.terminal_memory = terminal_mem
        agent.mem_cntr = mem_cntr
            
    scores, eps_history = [], []
    n_games = 50000

    time.sleep(10)

    for i in range(n_games):
        print("Game Number: " + str(i))
        score = 0
        done = False
        gameInput("0000000")
        data = sock.ReadReceivedData()
        while(data == None):
            data = sock.ReadReceivedData()
        while not done:
            if data == "DEAD":
                done = True
                break

            data_array = data.split(" ")
            curr_state = []

            for elem in data_array:
                curr_state.append(float(elem))

            if curr_state[6] <= 3.0:
                done = True
                break

            action = agent.choose_action(curr_state)
            #print("Action: " + str(action))
            binary_action = toBinary(action)
            while len(binary_action) < 6:
                binary_action = "0" + binary_action
            gameInput(binary_action)

            new_data = sock.ReadReceivedData()
            while new_data == None:
                new_data = sock.ReadReceivedData()

            #print("HERE")
            if new_data == "DEAD":
                done = True
                break

            new_data_array = new_data.split(" ")
            next_state = []

            for elem in new_data_array:
                next_state.append(float(elem))

            old_reward = calc_reward(curr_state[2], curr_state[3], curr_state[6], curr_state[7])
            reward = calc_reward(next_state[2], next_state[3], next_state[6], next_state[7])
            if old_reward > 0:
                reward -= old_reward
            else:
                reward += old_reward

            #print(reward)
            
            score += reward

            agent.store_transition(curr_state, action, reward, next_state, done)
            agent.learn()
            
            data = new_data
                    
        scores.append(score)
        eps_history.append(agent.epsilon)
        gameInput("0000000")
        
        if n_games % 100 == 0 and n_games > 0:
            with open("state_memory.txt", 'r+') as f:
                f.truncate(0)
            with open("new_state_memory.txt", 'r+') as f:
                f.truncate(0)
            with open("action_memory.txt", 'r+') as f:
                f.truncate(0)
            with open("reward_memory.txt", 'r+') as f:
                f.truncate(0)
            with open("terminal_memory.txt", 'r+') as f:
                f.truncate(0)
            with open("counter.txt", 'r+') as f:
                f.truncate(0)

            agent.state_memory.tofile('state_memory.txt', sep = " ", format = "%s")
            agent.new_state_memory.tofile('new_state_memory.txt', sep = " ", format = "%s")
            agent.action_memory.tofile('action_memory.txt', sep = " ", format = "%s")
            agent.reward_memory.tofile('reward_memory.txt', sep = " ", format = "%s")
            agent.new_state_memory.tofile('terminal_memory.txt', sep = " ", format = "%s")
            with open("counter.txt", 'r+') as f:
                mem_cntr = agent.mem_cntr
                f.write(str(mem_cntr))

