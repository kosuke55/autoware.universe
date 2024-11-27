import numpy as np
import torch
from torch import nn

acc_queue_size = 15
steer_queue_size = 15
prediction_step = 3

class GetInitialHiddenNN(nn.Module):
    def __init__(
        self,
        output_size,
        acc_hidden_sizes=(16,8),
        steer_hidden_sizes=(16,8),
        gru_hidden_size=32,
        key_size=32,
        value_size=32,
        num_heads=4,
        randomize=0.000001,
        mean_steps=3
    ):
        super(GetInitialHiddenNN, self).__init__()
        self.states_size = 3 # vel, acc, steer
        self.vel_index = 0
        self.acc_index = 1
        self.steer_index = 2
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads
        acc_input_indices = np.arange(self.states_size, self.states_size + acc_queue_size + prediction_step)
        steer_input_indices = np.arange(
            self.states_size + acc_queue_size + prediction_step, self.states_size + acc_queue_size + steer_queue_size + 2 * prediction_step
        )
        self.acc_layer_input_indices = np.concatenate(([self.vel_index,self.acc_index], acc_input_indices))
        self.steer_layer_input_indices = np.concatenate(([self.vel_index,self.steer_index], steer_input_indices))
        lb = -randomize
        ub = randomize
        self.acc_layer_1 = nn.Sequential(
            nn.Linear(len(self.acc_layer_input_indices), acc_hidden_sizes[0]),
            nn.ReLU()
        )
        nn.init.uniform_(self.acc_layer_1[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.acc_layer_1[0].bias, a=lb, b=ub)
        self.acc_layer_2 = nn.Sequential(
            nn.Linear(acc_hidden_sizes[0], acc_hidden_sizes[1]),
            nn.ReLU()
        )
        nn.init.uniform_(self.acc_layer_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.acc_layer_2[0].bias, a=lb, b=ub)
        self.steer_layer_1 = nn.Sequential(
            nn.Linear(len(self.steer_layer_input_indices), steer_hidden_sizes[0]),
            nn.ReLU()
        )
        nn.init.uniform_(self.steer_layer_1[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_1[0].bias, a=lb, b=ub)
        self.steer_layer_2 = nn.Sequential(
            nn.Linear(steer_hidden_sizes[0], steer_hidden_sizes[1]),
            nn.ReLU()
        )
        nn.init.uniform_(self.steer_layer_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_2[0].bias, a=lb, b=ub)
        combined_input_size = 1 + acc_hidden_sizes[1] + steer_hidden_sizes[1]
        self.gru = nn.GRU(
            input_size=combined_input_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(num_heads, key_size))
        nn.init.uniform_(self.query, a=lb, b=ub)
        self.key_layer = nn.Linear(gru_hidden_size, key_size * num_heads,bias=False) # bias is canceled by the softmax
        nn.init.uniform_(self.key_layer.weight, a=lb, b=ub)
        self.value_layer = nn.Linear(gru_hidden_size, value_size * num_heads)
        nn.init.uniform_(self.value_layer.weight, a=lb, b=ub)
        nn.init.uniform_(self.value_layer.bias, a=lb, b=ub)
        self.mean_steps = mean_steps
        self.final_layer = nn.Sequential(
            nn.Linear(value_size * num_heads, output_size),
            nn.Tanh()
        )
        nn.init.uniform_(self.final_layer[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.final_layer[0].bias, a=lb, b=ub)
    def forward(self, x, only_gru=False):
        batch_size, sample_size, seq_len, input_size = x.size()
        x_view = x.view(batch_size * sample_size, seq_len, input_size)
        acc_input = x_view[:, :, self.acc_layer_input_indices]
        steer_input = x_view[:, :, self.steer_layer_input_indices]
        acc_layer_1_out = self.acc_layer_1(acc_input)
        acc_layer_2_out = self.acc_layer_2(acc_layer_1_out)
        steer_layer_1_out = self.steer_layer_1(steer_input)
        steer_layer_2_out = self.steer_layer_2(steer_layer_1_out)
        combined_input = torch.cat((x_view[:, :, self.vel_index].unsqueeze(2), acc_layer_2_out, steer_layer_2_out), dim=2)
        gru_out = self.gru(combined_input)[0][:, -self.mean_steps:].mean(dim=1).view(batch_size, sample_size, -1)
        if only_gru:
            return gru_out
        key = self.key_layer(gru_out).view(batch_size, sample_size, self.num_heads, self.key_size)
        value = self.value_layer(gru_out).view(batch_size, sample_size, self.num_heads, self.value_size)
        attention = torch.einsum('bsnh,nh->bsn', key, self.query) / np.sqrt(self.key_size)
        attention = torch.softmax(attention, dim=1) # (batch_size, sample_size, num_heads)
        attention = attention.unsqueeze(3) # (batch_size, sample_size, num_heads, 1)
        context = torch.sum(attention * value, dim=1)
        context = context.view(batch_size, -1)
        return self.final_layer(context)

