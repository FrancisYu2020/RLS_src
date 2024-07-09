import torch
import torch.nn as nn

x = torch.randn(1, 128, 1792)
lstm = nn.LSTM(1792, 256, num_layers=2, dropout=0.1, bidirectional=True, batch_first=True)
output1, output2= lstm(x)
print(output1.shape, output2[0].shape, output2[1].shape)