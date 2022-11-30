
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

class Bi_LSTM(nn.Module):
    def __init__(self, num_hidden, input_dim, aux_dim, hand_dim):
        super(Bi_LSTM, self).__init__()
        self.num_hidden = num_hidden # no use
        self.input_dim = input_dim
        self.aux_dim = aux_dim
        self.hand_dim = hand_dim # no use

        self.bi_lstm1 = nn.LSTM(input_size  = self.input_dim + self.aux_dim,
                                hidden_size = 16,
                                num_layers = 1,
                                batch_first = True,
                                dropout = 0,
                                bidirectional = True)
        self.drop1 = nn.Dropout(p=0.2)

        self.bi_lstm2 = nn.LSTM(input_size  = 16,
                        hidden_size = 32,
                        num_layers = 1,
                        batch_first = True,
                        dropout = 0,
                        bidirectional = True)
        self.drop2 = nn.Dropout(p=0.2)

        self.fc = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(32, 16)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('drop1', nn.Dropout(p=0.2)),
                                ('fc2', nn.Linear(16, 16)),
                                ('relu2', nn.ReLU(inplace=True))
                                ]))

        # for mov
        self.bi_lstm3 = nn.LSTM(input_size  = self.input_dim*2,
                                hidden_size = 16,
                                num_layers = 1,
                                batch_first = True,
                                dropout = 0,
                                bidirectional = True)
        self.drop3 = nn.Dropout(p=0.2)

        self.bi_lstm4 = nn.LSTM(input_size  = 16,
                        hidden_size = 32,
                        num_layers = 1,
                        batch_first = True,
                        dropout = 0,
                        bidirectional = True)
        self.drop4 = nn.Dropout(p=0.2)

        self.fc2 = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(32, 16)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('drop1', nn.Dropout(p=0.2)),
                                ('fc2', nn.Linear(16, 16)),
                                ('relu2', nn.ReLU(inplace=True))
                                ]))

        self.drop5 = nn.Dropout(p=0.2)
        
        self.fc3 = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(16, 8)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ]))

        self.cls = nn.Linear(8, 1)

    # Defining the forward pass
    def forward(self, x, ops, mov):
        
        x_in = torch.cat((x, ops), dim = 2)

        x, hidden = self.bi_lstm1(x_in)
        x_split = torch.split(x, (x.shape[2]//2), 2)
        x = x_split[0] + x_split[1]
        x = self.drop1(x)
        x, hidden = self.bi_lstm2(x)
        x_split = torch.split(x, (x.shape[2]//2), 2)
        x = x_split[0] + x_split[1]
        x = self.drop2(x)

        x = x.split(1,1)[0]
        x_1 = x.reshape(x.shape[0], -1)
        x_1 = self.fc(x_1)       

        x_2, hidden = self.bi_lstm3(mov)
        x_2_split = torch.split(x_2, (x_2.shape[2]//2), 2)
        x_2 = x_2_split[0] + x_2_split[1]
        x_2 = self.drop3(x_2)
        x_2, hidden = self.bi_lstm4(x_2)
        x_2_split = torch.split(x_2, (x_2.shape[2]//2), 2)
        x_2 = x_2_split[0] + x_2_split[1]
        x_2 = self.drop4(x_2)

        x_2 = x_2.split(1,1)[0]
        x_2 = x_2.reshape(x_2.shape[0], -1)
        x_2 = self.fc2(x_2)   

        x_fuse = x_1 + x_2
        x4 = self.drop5(x_fuse)
        x4 = self.fc3(x4)

        out = self.cls(x4)

        return out