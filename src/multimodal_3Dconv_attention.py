import torch
import torch.nn as nn
from convlstm import ConvLSTM
# from mambapy.mamba import MambaConfig, Mamba

class ConvBlock(nn.Module):
    
    def __init__(self, channel_num, kernel_size, filters,strides=(1, 2, 2)):
        super(ConvBlock, self).__init__()

        self.channel_num = channel_num
        self.kernel_size = kernel_size
        self.filters = filters                              # numeri canali in uscita per ogni blocchetto
        self.strides = strides
        self.padding = tuple(k//2 for k in kernel_size)     # gestione padding dinamico per gestire diverse dimensioni kernel

        # 1. riduzione dimensionale
        # convoluzione 1x1 + batchNorm + ReLu
        # output channel dim = filters[0]
        # stride (1, 2, 2) = dimezza W e H
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels=self.channel_num, out_channels=self.filters[0], kernel_size=1, stride=self.strides, padding=0),
            nn.BatchNorm3d(self.filters[0]),
            nn.ReLU()
        )
        
        # 2. estrazione feature
        # convoluzione 3x3 + batchNorm + ReLu
        # output channel dim = filters[1]
        # stride default (1, 1, 1), dimesioni H e W non cambiano
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm3d(self.filters[1]),
            nn.ReLU()
        )

        # 3. espanzione canali
        # convoluzione 1x1 + batchNorm 
        # output channel dim = filters[2]
        # no relu perche output viene sommato a shortcut, toglierei i valori negativi prima della somma
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(in_channels=self.filters[1], out_channels=self.filters[2], kernel_size=1, padding=0),
            nn.BatchNorm3d(self.filters[2])
        )

        # shortcut da input originale (vedi forward)
        # per sommare output blocco 3 e x iniziale serve che abbiano la stessa dimensione, stride su x per dimezzare W e H
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels=self.channel_num, out_channels=self.filters[2], kernel_size=1, stride=self.strides, padding=0),
            nn.BatchNorm3d(self.filters[2])
        )

        self.relu = nn.ReLU()
	
    # comportamento del blocco con input tensore x
    # residual = tensore x 5D (batch, canali, tempo, H, W)
    # esegue blocco 1, 2 e 3
    # somma skipconnection
    # relu finale
    # output = tensore 5D (batch, filters[2], tempo, H/2, W/2)
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
        self.output_padding = tuple(s-1 for s in self.strides)  # per avere padding perfetto per tornare a dimensioni originali corrette

        # convTranspose3d = convoluzione trasposta, stride per aumentare dimensioni invece che diminuire
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
        self.hid_channel = in_channel // reduction_ratio    # [in realtà gli istanti temporali] non analizza i canali contemporaneamente ma li divide per un fattore reduction_ratio
        self.dilation = dilation

        # blocco 3D (H, W, t) in un singolo numero per canale
        self.globalAvgPool = nn.AdaptiveAvgPool3d(1)    # media attivazione
        self.globalMaxPool = nn.AdaptiveMaxPool3d(1)    # massimo

        # MLP capisce quali canali sono importanti e quali rumore
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=self.hid_channel),   # riduce i canali in hid_channel
            nn.ReLU(),
            nn.Linear(in_features=self.hid_channel, out_features=in_channel)    # riaumenta a canali originali (se no Mc*x non si può fare)
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # convoluzione 7x7, usata dopo per trovare pixel importanti
        self.conv1 = nn.Conv2d(2, 1, kernel_size=(7, 7), stride=1, padding=3, dilation=self.dilation, bias=False)

    def forward(self, x):
        # cosa guardare: importanza ciascun istante temporale
        # passa valore medio a MLP
        # traspose inverte i canali e tempo, avg e max prendono gli ultimi tre valori e calcolano media e max
        # senza inversione prende t, H, W e gli schiaccia creando un numero per ogni canale
        # 
        avgOut = self.globalAvgPool(torch.transpose(x, 1, 2))       
        avgOut = avgOut.view(avgOut.size(0), -1)
        avgOut = self.mlp(avgOut)

        # passa valore massimo a MLP
        maxOut = self.globalMaxPool(torch.transpose(x, 1, 2))
        maxOut = maxOut.view(maxOut.size(0), -1)
        maxOut = self.mlp(maxOut)

        # somma valori e applica sigmoid (percentuali da 0 a 1)
        # moltiplica Mc (vettore percentuale per istante) con x: identifica canali utili
        # Mc: (Batch, 1, Tempo, 1, 1) * x: (Batch, Canali, Tempo, Altezza, Larghezza)
        Mc = self.sigmoid(avgOut + maxOut)
        Mc = Mc.view(Mc.size(0), 1, Mc.size(1), 1, 1)
        Mf1 = Mc * x

        # dove guardare: mappa di calore
        # schiaccia tutti i canali e tempo di Mf1 in una immagine 2D Ms

        # GIA COMMENTATO: sigmoid(conv7x7( [AvgPool(F); MaxPool(F)]))
        # unsqeeze schiaccia 1=canali e 2=tempo -> diventa (batch, 1, H, W)
        # cat = concatenazione canali -> Ms = (batch, 2, H, W)
        maxOut = torch.amax(Mf1, (1, 2)).unsqueeze(1)   
        avgOut = torch.mean(Mf1, (1, 2)).unsqueeze(1)
        Ms = torch.cat((maxOut, avgOut), dim=1)

        # mappa attraverso conv2D(7x7): prende mappa max e mappa avg e fa concoluzione per generare mappa unica
        # sigmoid = Ms mappa di attenzione (pixel importanti 1, statici 0)
        # moltiplica Ms e Mf1 (istanti temporali utili)
        # ritorna Mf2: tensore pulito da rumore con focus su istanti temporali importanti
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

        # 6+1 layers: si usa ConvBlock e CBAM definiti prima

        # stage 0. prima scansione (1x7x7), tempo congelato = considera un t alla volta
        # input dim = canali satellite
        # output channel = 16
        # dimensioni H e W invariate
        self.conv1 = nn.Sequential(nn.Conv3d(input_dim, 16, kernel_size=(1,7,7), padding='same', stride=(1, 1, 1)),
								  nn.BatchNorm3d(16),
								  nn.ReLU()
		)

        # stage 1 - 2 - 3 - 4 - 5 - 6. ConvBlock + CBAM
        # CBAM n_images = istanti temporali serie
        # output channel = 256
        # dimensioni H = 4, W = 4
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

        # Mamba per analisi temporale
        if self.mamba:
            self.ln = nn.LayerNorm([256*4*4])
            self.mamba_proj_in = nn.Linear(256*4*4, 256)
            self.mamba_config = MambaConfig(d_model=256,n_layers=1)
            self.mamba = Mamba(self.mamba_config)
            self.fc = nn.Linear(256*n_images, output_dim)

        # ConvLSTM per analisi temporale
        # layer lineare finale fully connected, trasforma vettore in output_dim
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
        x = x.permute(0, 2, 1, 3, 4)  # cambio tensore 5D, da (batch, channels, time, height, width) passo a (batch, time, channels, height, width) necessario per passare x a LSTM o Mamba

        if self.mamba:
            x = x.reshape(x.size(0), x.size(1), -1)  
            x = self.ln(x)
            x = self.mamba_proj_in(x)  # Project to mamba input dimension
            x = self.mamba(x)  # Apply Mamba
            x = torch.flatten(x, start_dim=1)  # Flatten the output

        else:
            x, _ = self.conv_lstm(x)        # lstm genera due output tensore e hidden state delle celle lstm. ritorna solo in tensore x,_  x[0]
            x = x[0]                        # il tensore è 5D (batch, time=4, channels=64, H=4, W=4)
            x = x.permute(0, 2, 1, 3, 4)    # inverte canali - tempo
            x = x.reshape(x.size(0), -1)    # flattening layer (dim 4096)
        x = self.fc(x)                      # layer fully connected (dim = output_dim)
        return x


class Multimodal_Encoder(nn.Module):
    def __init__(self, input_dim1=2, input_dim2=1, output_dim=256, n_images1=10, n_images2=20, device='cuda'):
        super(Multimodal_Encoder, self).__init__()
        self.device = device

        # stessa struttura di SingleModal Encoder
        # _1 = sar, _2 = ottico
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
        self.stage5_2 = nn.Sequential(ConvBlock(256, kernel_size=(1,3,3), filters=[64, 64, 256], strides=(1, 2, 2)),
                                  CBAM(n_images2//2, reduction_ratio=4)
        )

        self.conv_lstm = ConvLSTM(input_dim=512, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(640, output_dim)

    # x in questo caso non è un tensore è un dizionario
    # contiene sia il tensore sar che ottico
    # .float() perchè dati tiff sono interi
    def forward(self, x):
        x1 = x['im1'].float()
        x1 = self.conv1_1(x1)
        x1 = self.stage1_1(x1)
        x1 = self.stage2_1(x1)
        x1 = self.stage3_1(x1)
        x1 = self.stage4_1(x1)
        x1 = self.stage5_1(x1)
        x1 = x1.permute(0, 2, 1, 3, 4)  # (batch, time, channels, height, width)

        x2 = x['im2'].float()#.to(non_blocking=True, device=self.device)
        x2 = self.conv1_2(x2)
        x2 = self.stage1_2(x2)
        x2 = self.stage2_2(x2)
        x2 = self.stage3_2(x2)
        x2 = self.stage4_2(x2)
        x2 = self.stage5_2(x2)
        x2 = x2.permute(0, 2, 1, 3, 4)  # (batch, time, channels, height, width)

        # x diventa tensore e concatenazione canali x1 e x2 (256 + 256 = 512)
        # si passa x a conv_lstm
        x = torch.cat((x1, x2), dim=2)   
        x, _ = self.conv_lstm(x)
        x = x[0] 

        # fonde tempo e canali insieme (4*64)
        # avg spreme immagine  4x4 in 1x1 (batch, 4*64, 1, 1), non lo faceva in singleEcoder, qui ho il doppio dei dati, vettore troppo lungo
        # si usa view invece di permute perche ho fatto avg, che in singleEncoder non c'era
        # layer flatten = vettore dim 4*64
        # fully connecteg layer = output_dim
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))  
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
          
                
    # x passa dal singleEncoder
    # si rifà fc per ripristinare le dimensioni di prima (pre fc singleEncoder)
    # è il primo passaggio di ricorstruzione
    def forward(self, x):
        x = x.float()
        x = self.encoder(x)
        x = self.fc(x)

        if self.mamba:
            x = x.reshape(x.size(0), self.n_images, 256)        # Reshape to (batch, time, features)
            x = self.mamba(x)                                   # Apply Mamba
            x = self.mamba_proj_out(x)                          # Project back to spatial dimensions
            x = x.reshape(x.size(0), self.n_images, 256, 4, 4)  # Reshape to (batch, channels, time, height, width)

        else:
            x = x.reshape(x.size(0), self.n_images, 64, 4, 4)   # (batch, time, channels, height, width)
            x, _ = self.conv_lstm(x)
            x = x[0]  

        x = x.permute(0, 2, 1, 3, 4)                            # (batch, channels, time, height, width)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.conv1(x)
        return x


# DOMANDE
#
# 1. CBAM channel_in -> canali o tempo?
# 2. CBAM reduction_ratio -> solo 4 istanti temporali, serve?
# 3. Singlemodal_CAE mai usato
# 4. Multimodal_Encoder mai usato
#
