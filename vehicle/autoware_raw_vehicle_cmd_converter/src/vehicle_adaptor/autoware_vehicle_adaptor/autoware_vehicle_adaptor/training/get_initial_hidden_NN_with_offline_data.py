import numpy as np
import torch
from torch import nn

acc_queue_size = 15
steer_queue_size = 15
prediction_step = 3

class Preprocess(nn.Module):
    def __init__(
        self,
        acc_hidden_sizes=(16,8),
        steer_hidden_sizes=(16,8),
        randomize=0.01
    ):
        super(Preprocess, self).__init__()
        self.states_size = 3 # vel, acc, steer
        self.vel_index = 0
        self.acc_index = 1
        self.steer_index = 2
        acc_input_indices = np.arange(self.states_size, self.states_size + acc_queue_size + prediction_step)
        steer_input_indices = np.arange(
            self.states_size + acc_queue_size + prediction_step, self.states_size + acc_queue_size + steer_queue_size + 2 * prediction_step
        )
        self.acc_layer_input_indices = np.concatenate(([self.vel_index,self.acc_index], acc_input_indices))
        self.steer_layer_input_indices = np.concatenate(([self.vel_index,self.steer_index], steer_input_indices))
        lb = -randomize
        ub = randomize
        # Before the GRU
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
        self.combined_input_size = 1 + acc_hidden_sizes[1] + steer_hidden_sizes[1]

    def forward(self, x):
        acc_input = x[:, :, self.acc_layer_input_indices]
        acc_layer_1_out = self.acc_layer_1(acc_input)
        acc_layer_2_out = self.acc_layer_2(acc_layer_1_out)
        steer_input = x[:, :, self.steer_layer_input_indices]
        steer_layer_1_out = self.steer_layer_1(steer_input)
        steer_layer_2_out = self.steer_layer_2(steer_layer_1_out)
        combined_input = torch.cat((x[:, :, self.vel_index].unsqueeze(2), acc_layer_2_out, steer_layer_2_out), dim=2)
        return combined_input
class GetOfflineFeatures(nn.Module):
    def __init__(
        self,
        input_size, # combined_input_size
        output_size, # 2 * lstm_encoder_hidden_size
        gru_hidden_size=32,
        key_size=32,
        value_size=32,
        num_heads=4,
        randomize=0.01,
        mean_steps=3
    ):
        super(GetOfflineFeatures, self).__init__()
        lb = -randomize
        ub = randomize
        input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.gru_offline = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True
        )
        nn.init.uniform_(self.gru_offline.weight_ih_l0, a=lb, b=ub)
        nn.init.uniform_(self.gru_offline.weight_hh_l0, a=lb, b=ub)
        nn.init.uniform_(self.gru_offline.bias_ih_l0, a=lb, b=ub)
        nn.init.uniform_(self.gru_offline.bias_hh_l0, a=lb, b=ub)
        self.query_offline = nn.Parameter(torch.randn(num_heads, key_size))
        nn.init.uniform_(self.query_offline, a=lb, b=ub)
        self.key_layer_offline = nn.Linear(gru_hidden_size, key_size * num_heads,bias=False) # bias is canceled by the softmax
        nn.init.uniform_(self.key_layer_offline.weight, a=lb, b=ub)
        self.value_layer_offline = nn.Linear(gru_hidden_size, value_size * num_heads)
        nn.init.uniform_(self.value_layer_offline.weight, a=lb, b=ub)
        nn.init.uniform_(self.value_layer_offline.bias, a=lb, b=ub)
        self.mean_steps = mean_steps
        self.final_layer_offline = nn.Sequential(
            nn.Linear(value_size * num_heads, output_size),
            nn.Tanh()
        )
        nn.init.uniform_(self.final_layer_offline[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.final_layer_offline[0].bias, a=lb, b=ub)
    def forward(self, x, preprocessor):
        batch_size, sample_size, seq_len, input_size = x.size()
        x_view = x.view(batch_size * sample_size, seq_len, input_size)
        combined_input = preprocessor(x_view)
        gru_out = self.gru_offline(combined_input)[0][:, -self.mean_steps:].mean(dim=1).view(batch_size, sample_size, -1)
        key = self.key_layer_offline(gru_out).view(batch_size, sample_size, self.num_heads, self.key_size)
        value = self.value_layer_offline(gru_out).view(batch_size, sample_size, self.num_heads, self.value_size)
        attention = torch.einsum('bsnh,nh->bsn', key, self.query_offline) / np.sqrt(self.key_size)
        attention = torch.softmax(attention, dim=1) # (batch_size, sample_size, num_heads)
        attention = attention.unsqueeze(3) # (batch_size, sample_size, num_heads, 1)
        context = torch.sum(attention * value, dim=1)
        context = context.view(batch_size, -1)
        return self.final_layer_offline(context)

        
class GetInitialHiddenNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        gru_hidden_size=32,
        #online_attn_hidden_size=32,
        randomize=0.01,
    ):
        super(GetInitialHiddenNN, self).__init__()
        lb = -randomize
        ub = randomize
        # online data
        self.gru_online = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True
        )
        nn.init.uniform_(self.gru_online.weight_ih_l0, a=lb, b=ub)
        nn.init.uniform_(self.gru_online.weight_hh_l0, a=lb, b=ub)
        nn.init.uniform_(self.gru_online.bias_ih_l0, a=lb, b=ub)
        nn.init.uniform_(self.gru_online.bias_hh_l0, a=lb, b=ub)
        """
        self.attn_online = nn.Sequential(
            nn.Linear(gru_hidden_size, online_attn_hidden_size),
            nn.Tanh(),
            nn.Linear(online_attn_hidden_size, 1)
        )
        nn.init.uniform_(self.attn_online[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.attn_online[0].bias, a=lb, b=ub)
        nn.init.uniform_(self.attn_online[2].weight, a=lb, b=ub)
        nn.init.uniform_(self.attn_online[2].bias, a=lb, b=ub)
        """
        self.final_layer_online = nn.Sequential(
            nn.Linear(gru_hidden_size, output_size),
            nn.Tanh()
        )

        # fusion case
        self.transform = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.Tanh()
        )
        nn.init.uniform_(self.transform[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.transform[0].bias, a=lb, b=ub)

        # only online case
        # self.only_online = nn.Sequential(
        #     nn.Linear(output_size, output_size),
        #     nn.Tanh()
        # )
        # nn.init.uniform_(self.only_online[0].weight, a=lb, b=ub)
        # nn.init.uniform_(self.only_online[0].bias, a=lb, b=ub)
    def forward(self, offline_out, mask=None): #online_x, mask, preprocessor):
        """
        mask: Tensor of shape (batch_size, 1), 1 if offline data is available, 0 otherwise
        """
        if mask is None:
            return self.transform(offline_out)
        else:
            mask = mask.float().unsqueeze(1)
            return mask * self.transform(offline_out)
        #online_combined_input = preprocessor(online_x)
        #online_gru_out = self.gru_online(online_combined_input)[1][0]
        #online_gru_out = self.gru_online(online_combined_input)[0]
        #online_attn = self.attn_online(online_gru_out) # (batch_size, seq_len, 1)
        #online_attn = torch.softmax(online_attn, dim=1)
        #online_context = torch.sum(online_attn * online_gru_out, dim=1)
        #online_out = self.final_layer_online(online_gru_out)

        #mask = mask.float().unsqueeze(1)
        #fused_out = self.fusion(torch.cat((offline_out, online_out), dim=1))
        #only_online_out = self.only_online(online_out)
        # print(mask.shape, fused_out.shape, only_online_out.shape)
        # return mask * fused_out + (1 - mask) * only_online_out       

