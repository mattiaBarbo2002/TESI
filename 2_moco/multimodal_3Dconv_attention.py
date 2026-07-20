import torch
import torch.nn as nn
from convlstm import ConvLSTM
from mambapy.mamba import MambaConfig, Mamba

class ConvBlock(nn.Module):
    def __init__(self, channel_num, kernel_size, filters,strides=(1, 2, 2)):
        super(ConvBlock, self).__init__()

        self.channel_num = channel_num
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = tuple(k//2 for k in kernel_size)

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels=self.channel_num, out_channels=self.filters[0], kernel_size=1, stride=self.strides, padding=0),
            nn.BatchNorm3d(self.filters[0]),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm3d(self.filters[1]),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(in_channels=self.filters[1], out_channels=self.filters[2], kernel_size=1, padding=0),
            nn.BatchNorm3d(self.filters[2])
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels=self.channel_num, out_channels=self.filters[2], kernel_size=1, stride=self.strides, padding=0),
            nn.BatchNorm3d(self.filters[2])
        )
        self.relu = nn.ReLU()
	
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x + residual
        out = self.relu(x)
        return out

class DeconvBlock(nn.Module):
    def __init__(self, channel_num, kernel_size, filters,strides=(1, 2, 2)):
        super(DeconvBlock, self).__init__()

        self.channel_num = channel_num
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = tuple(k//2 for k in kernel_size)
        self.output_padding = tuple(s-1 for s in self.strides)

        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.channel_num, out_channels=self.filters[0], kernel_size=1, stride=self.strides, padding=0,output_padding=self.output_padding),
            nn.BatchNorm3d(self.filters[0]),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm3d(self.filters[1]),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.filters[1], out_channels=self.filters[2], kernel_size=1, padding=0),
            nn.BatchNorm3d(self.filters[2])
        )
        self.shortcut = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.channel_num, out_channels=self.filters[2], kernel_size=1, stride=self.strides, padding=0,output_padding=self.output_padding),
            nn.BatchNorm3d(self.filters[2])
        )
        self.relu = nn.ReLU()
	
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x + residual
        out = self.relu(x)
        return out

class CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio, dilation=1):
        super(CBAM, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation

        self.globalAvgPool = nn.AdaptiveAvgPool3d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool3d(1)

        # Shared MLP.
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=self.hid_channel),
            nn.ReLU(),
            nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=(7, 7), stride=1, padding=3, dilation=self.dilation, bias=False)

    def forward(self, x):
        ''' Channel attention '''
        avgOut = self.globalAvgPool(torch.transpose(x, 1, 2))
        avgOut = avgOut.view(avgOut.size(0), -1)
        avgOut = self.mlp(avgOut)

        maxOut = self.globalMaxPool(torch.transpose(x, 1, 2))
        maxOut = maxOut.view(maxOut.size(0), -1)
        maxOut = self.mlp(maxOut)
        Mc = self.sigmoid(avgOut + maxOut)
        Mc = Mc.view(Mc.size(0), 1, Mc.size(1), 1, 1)
        Mf1 = Mc * x

        ''' Spatial attention. '''
        # sigmoid(conv7x7( [AvgPool(F); MaxPool(F)]))
        maxOut = torch.amax(Mf1, (1, 2)).unsqueeze(1)
        avgOut = torch.mean(Mf1, (1, 2)).unsqueeze(1)
        Ms = torch.cat((maxOut, avgOut), dim=1)

        Ms = self.conv1(Ms)
        Ms = self.sigmoid(Ms)
        Ms = Ms.view(Ms.size(0), 1, 1, Ms.size(2), Ms.size(3))
        Mf2 = Ms * Mf1
        return Mf2

class Singlemodal_Encoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=256, n_images=10, mamba=False):
        super(Singlemodal_Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_images = n_images
        self.mamba = mamba
        self.conv1 = nn.Sequential(nn.Conv3d(input_dim, 16, kernel_size=(1,7,7), padding='same', stride=(1, 1, 1)),
								  nn.BatchNorm3d(16),
								  nn.ReLU()
		)
        self.stage1 = nn.Sequential(ConvBlock(16, kernel_size=(1,3,3), filters=[16, 16, 64], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)
        self.stage2 = nn.Sequential(ConvBlock(64, kernel_size=(1,3,3), filters=[16, 16, 64], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)
        self.stage3 = nn.Sequential(ConvBlock(64, kernel_size=(1,3,3), filters=[32, 32, 128], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)
        self.stage4 = nn.Sequential(ConvBlock(128, kernel_size=(1,3,3), filters=[32, 32, 128], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)
        self.stage5 = nn.Sequential(ConvBlock(128, kernel_size=(1,3,3), filters=[64, 64, 256], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)
        self.stage6 = nn.Sequential(ConvBlock(256, kernel_size=(1,3,3), filters=[64, 64, 256], strides=(1, 2, 2)),
                                    CBAM(n_images, reduction_ratio=4)
        )
        if self.mamba:
            self.ln = nn.LayerNorm([256*4*4])
            self.mamba_proj_in = nn.Linear(256*4*4, 256)
            self.mamba_config = MambaConfig(d_model=256,n_layers=1)
            self.mamba = Mamba(self.mamba_config)
            self.fc = nn.Linear(256*n_images, output_dim)
        else:
            self.conv_lstm = ConvLSTM(input_dim=256, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
            self.fc = nn.Linear(1024*n_images, output_dim)

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = x.permute(0, 2, 1, 3, 4)  # Rearrange to (batch, time, channels, height, width)
        if self.mamba:
            x = x.reshape(x.size(0), x.size(1), -1)  # Flatten the spatial dimensions
            x = self.ln(x)
            x = self.mamba_proj_in(x)  # Project to mamba input dimension
            x = self.mamba(x)  # Apply Mamba
            x = torch.flatten(x, start_dim=1)  # Flatten the output
        else:
            x, _ = self.conv_lstm(x)
            x = x[0]  # Get the output from the ConvLSTM
            x = x.permute(0, 2, 1, 3, 4)  # Rearrange to (batch, channels, time, height, width)
            x = x.reshape(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Fully connected layer for output
        return x

class Multimodal_Encoder(nn.Module):
    def __init__(self, input_dim1=2, input_dim2=1, output_dim=256, n_images1=10, n_images2=20, device='cuda'):
        super(Multimodal_Encoder, self).__init__()
        self.device = device
        self.conv1_1 = nn.Sequential(nn.Conv3d(input_dim1, 16, kernel_size=(1,7,7), padding='same', stride=(1, 1, 1)),
								  nn.BatchNorm3d(16),
								  nn.ReLU()
		)
        self.conv1_2 = nn.Sequential(nn.Conv3d(input_dim2, 16, kernel_size=(1,7,7), padding='same', stride=(1, 1, 1)),
								  nn.BatchNorm3d(16),
								  nn.ReLU()
		)
        self.stage1_1 = nn.Sequential(ConvBlock(16, kernel_size=(1,3,3), filters=[16, 16, 64], strides=(1, 2, 2)),
								  CBAM(n_images1, reduction_ratio=4)
		)
        self.stage1_2 = nn.Sequential(ConvBlock(16, kernel_size=(1,3,3), filters=[16, 16, 64], strides=(1, 2, 2)),
								  CBAM(n_images2, reduction_ratio=4)
		)
        self.stage2_1 = nn.Sequential(ConvBlock(64, kernel_size=(1,3,3), filters=[32, 32, 128], strides=(1, 2, 2)),
								  CBAM(n_images1, reduction_ratio=4)
		)
        self.stage2_2 = nn.Sequential(ConvBlock(64, kernel_size=(1,3,3), filters=[32, 32, 128], strides=(1, 2, 2)),
								  CBAM(n_images2, reduction_ratio=4)
		)
        self.stage3_1 = nn.Sequential(ConvBlock(128, kernel_size=(1,3,3), filters=[32, 32, 128], strides=(1, 2, 2)),
								  CBAM(n_images1, reduction_ratio=4)
		)
        self.stage3_2 = nn.Sequential(ConvBlock(128, kernel_size=(1,3,3), filters=[32, 32, 128], strides=(1, 2, 2)),
                                  CBAM(n_images2, reduction_ratio=4)
        )
        self.stage4_1 = nn.Sequential(ConvBlock(128, kernel_size=(1,3,3), filters=[64, 64, 256], strides=(1, 2, 2)),
								  CBAM(n_images1, reduction_ratio=4)
		)
        self.stage4_2 = nn.Sequential(ConvBlock(128, kernel_size=(1,3,3), filters=[64, 64, 256], strides=(1, 2, 2)),
                                  CBAM(n_images2, reduction_ratio=4)
        )
        self.stage5_1 = nn.Sequential(ConvBlock(256, kernel_size=(1,3,3), filters=[64, 64, 256], strides=(1, 2, 2)),
								  CBAM(n_images1, reduction_ratio=4)
		)
        self.stage5_2 = nn.Sequential(ConvBlock(256, kernel_size=(1,3,3), filters=[64, 64, 256], strides=(2, 2, 2)),
                                  CBAM(n_images2//2, reduction_ratio=4)
        )
        self.conv_lstm = ConvLSTM(input_dim=512, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(640, output_dim)

    def forward(self, x):
        x1 = x['im1'].float()#.to(non_blocking=True, device=self.device)
        x1 = self.conv1_1(x1)
        x1 = self.stage1_1(x1)
        x1 = self.stage2_1(x1)
        x1 = self.stage3_1(x1)
        x1 = self.stage4_1(x1)
        x1 = self.stage5_1(x1)
        x1 = x1.permute(0, 2, 1, 3, 4)  # Rearrange to (batch, time, channels, height, width)
        x2 = x['im2'].float()#.to(non_blocking=True, device=self.device)
        x2 = self.conv1_2(x2)
        x2 = self.stage1_2(x2)
        x2 = self.stage2_2(x2)
        x2 = self.stage3_2(x2)
        x2 = self.stage4_2(x2)
        x2 = self.stage5_2(x2)
        x2 = x2.permute(0, 2, 1, 3, 4)  # Rearrange to (batch, time, channels, height, width)
        x = torch.cat((x1, x2), dim=2)  # Concatenate along the channel dimension
        x, _ = self.conv_lstm(x)
        x = x[0]  # Get the output from the ConvLSTM
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))  # Merge time and channel dimensions
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Fully connected layer for output
        return x

class Singlemodal_CAE(nn.Module):
    def __init__(self, input_dim=2, output_dim=256, n_images=10, mamba=False):
        super(Singlemodal_CAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_images = n_images
        self.mamba = mamba
        self.encoder = Singlemodal_Encoder(input_dim=input_dim, output_dim=output_dim, n_images=n_images, mamba=mamba)
        if self.mamba:
            self.fc = nn.Linear(output_dim,256*n_images)
            self.mamba_config = MambaConfig(d_model=256,n_layers=1)
            self.mamba = Mamba(self.mamba_config)
            self.mamba_proj_out = nn.Linear(256, 256*4*4)
        else:
            self.fc = nn.Linear(output_dim,1024*n_images)
            self.conv_lstm = ConvLSTM(input_dim=64, hidden_dim=256, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.stage1 = nn.Sequential(DeconvBlock(256, kernel_size=(1,3,3), filters=[64, 64, 256], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)
        self.stage2 = nn.Sequential(DeconvBlock(256, kernel_size=(1,3,3), filters=[64, 64, 256], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)
        self.stage3 = nn.Sequential(DeconvBlock(256, kernel_size=(1,3,3), filters=[32, 32, 128], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)
        self.stage4 = nn.Sequential(DeconvBlock(128, kernel_size=(1,3,3), filters=[32, 32, 128], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)
        self.stage5 = nn.Sequential(DeconvBlock(128, kernel_size=(1,3,3), filters=[16, 16, 64], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)    
        self.stage6 = nn.Sequential(DeconvBlock(64, kernel_size=(1,3,3), filters=[16, 16, 64], strides=(1, 2, 2)),
								  CBAM(n_images, reduction_ratio=4)
		)                         
        self.conv1 = nn.Conv3d(64, input_dim, kernel_size=(1,7,7), padding='same', stride=(1, 1, 1))
          
                
    def forward(self, x):
        x = x.float()
        x = self.encoder(x)
        x = self.fc(x)
        if self.mamba:
            x = x.reshape(x.size(0), self.n_images, 256)  # Reshape to (batch, time, features)
            x = self.mamba(x)  # Apply Mamba
            x = self.mamba_proj_out(x)  # Project back to spatial dimensions
            x = x.reshape(x.size(0), self.n_images, 256, 4, 4)  # Reshape to (batch, channels, time, height, width)
        else:
            x = x.reshape(x.size(0), self.n_images, 64, 4, 4)  # Reshape to (batch, channels, time, height, width)
            x, _ = self.conv_lstm(x)
            x = x[0]  # Get the output from the ConvLSTM
        x = x.permute(0, 2, 1, 3, 4)  # Rearrange to (batch, channels, time, height, width)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.conv1(x)
        return x