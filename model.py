from torch import nn
import torch

class LSTMNet(nn.Module):
    
    def __init__(self, emb_size=26, hidden_size=1024, lstm_layers=1, dropout=0.1):
        super(LSTMNet, self).__init__()
        self.emb_size = emb_size
        self.layers = lstm_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, emb_size)
        
        
    def forward(self, x, pv_st):
        out, st = self.lstm(x, pv_st)
        out=self.fc1(out)
        return out, st

    def init_state(self, batch_size=None):
        if batch_size:
            return (
                torch.zeros((self.layers, batch_size, self.hidden_size), dtype=torch.float),
                torch.zeros((self.layers, batch_size, self.hidden_size),dtype=torch.float)
            )
        else:
            return (
                torch.zeros((self.layers, self.hidden_size), dtype=torch.float),
                torch.zeros((self.layers, self.hidden_size),dtype=torch.float)
            )

