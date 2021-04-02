import torch
import torch.nn as nn
import torch.nn.functional as F

class FmriModel(nn.Module):
    
    def __init__(self, params):
        super(FmriModel, self).__init__()
        
        # Using only one convolutional layer
        self.conv = nn.Sequential(
            nn.Conv2d(params.nX, params.conv_channels, kernel_size=3, stride=1, bias=False)
        )
        
        # Extra work to automatically find sizes for conv output and lstm input
        self._to_linear, self._to_lstm = None, None
        # e.g. A batch would be of size: (b_s, 85, 57, 68, 49)
        x = torch.randn(params.batch_size, params.seg_len, params.nX, params.nY, params.nZ)
        # An initial pass to find the output size from conv layer
        self.convs(x)
        
        # LSTM layer for the sequence input
        self.lstm = nn.LSTM(input_size=self._to_lstm, hidden_size=params.rnn_hidden_size,
                           num_layers=1, batch_first=True)
        
        # One FC layer for classification
        self.fc = nn.Linear(params.rnn_hidden_size, params.num_classes)
        
    def convs(self, x):
        batch_size, timesteps, c, h, w = x.size()
        x = x.view(batch_size*timesteps, c, h, w)
        x = self.conv(x)
        
        if self._to_linear is None:
            # Just runs during model initialization to calculate output size of conv layer
            self._to_linear = int(x[0].shape[0]*x[0].shape[1]*x[0].shape[2])
            r_in = x.view(batch_size, timesteps, -1)
            self._to_lstm = r_in.shape[2]
            
        return x
    
    def forward(self, x):
        batch_size, timesteps, c, h, w = x.size()
        cnn_out = self.convs(x)

        # Prepare the output from CNN to pass through the LSTM layer
        r_in = cnn_out.view(batch_size, timesteps, -1)

        # Flattening is required when we use DataParallel
        self.lstm.flatten_parameters()

        # Get output from the LSTM
        r_out, (h_n, h_c) = self.lstm(r_in)

        # Pass the output of the LSTM to FC layers
        r_out = self.fc(r_out[:, -1, :])

        # Apply softmax to the output and return it
        return F.log_softmax(r_out, dim=1)