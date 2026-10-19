import copy
import numpy as np
import torch
import torch.nn as nn

"""

lightweight-temporal-attention from https://doi.org/10.48550/arXiv.2107.07933

github: https://github.com/VSainteuf/utae-paps.git


"""


# input stage 6 dopo permute -> (batch, T=4, C=256, H=64, W=64) 
"""
    per ogni pixel della mappa di feature ho 4 (istanti) vettori lunghi 256 -> vec1, vec2, vec3, vec4
    ogni pixel è il risultato dei 6 layer convoluzionali 

    QUERY: cosa sto cercando, deciso a priori

    KEY: 
"""


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table

"""
    d_model = 256 -> passo direttamente i canali
    T = 1000
    offset = 0

    

"""



class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn



class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


# codice nuovo spatial attention


class SpatialAttn2d(nn.Module):
    def __init__(
        self,
        in_channels=256,
        n_head=8,
        d_k=8,
        mlp=[256, 256],
        dropout=0.2,
        d_model=None,
        sr_ratio=4,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        """
        Spatial Self-Attention per serie di immagini.
        Ogni pixel si confronta con tutte le regioni dell'immagine, separatamente per ogni istante.
        A differenza di LTAE la sequenza non viene schiacciata: output con la stessa shape dell'input.
        Args:
            in_channels (int): Numero di canali dell'input.
            n_head (int): Numero di head di attention.
            d_k (int): Dimensione dei vettori key e query.
            mlp (List[int]): Larghezze dei layer dell'MLP che elabora le head concatenate.
                mlp[0] deve essere d_model, mlp[-1] deve essere in_channels (per la connessione residua).
            dropout (float): dropout
            d_model (int, optional): Se specificato, l'input viene prima proiettato in d_model canali.
            sr_ratio (int): Fattore di spatial reduction per key/value (4 -> 64x64 diventa 16x16).
            T (int): Periodo del positional encoding.
            return_att (bool): Se True, ritorna anche le maschere di attention.
            positional_encoding (bool): Se False, nessun positional encoding.
        """
        super(SpatialAttn2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head
        self.sr_ratio = sr_ratio

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model
        assert self.mlp[-1] == self.in_channels      # DIFF: serve per sommare l'output all'input

        # DIFF: positional encoding 2D -> metà canali per la riga, metà per la colonna
        if positional_encoding:
            self.positional_encoder = PositionalEncoder(self.d_model // 2, T=T)
        else:
            self.positional_encoder = None

        # DIFF: spatial reduction per key/value
        if sr_ratio > 1:
            self.sr = nn.Conv2d(self.d_model, self.d_model, kernel_size=sr_ratio, stride=sr_ratio)
        else:
            self.sr = None

        self.attention_heads = MultiHeadSpatialAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        # DIFF: connessione residua, parte da 0 -> all'inizio il blocco è un'identità
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # DIFF: input (B, C, T, H, W) come gli stage dell'encoder (LTAE vuole (B, T, C, H, W))
        sz_b, d, seq_t, h, w = x.shape

        # DIFF: T nel batch, H*W è la sequenza (in LTAE: H, W nel batch, T è la sequenza)
        out = x.permute(0, 2, 1, 3, 4).contiguous().view(sz_b * seq_t, d, h * w)   # (B*T, C, H*W)
        out = self.in_norm(out)

        if self.inconv is not None:
            out = self.inconv(out)                                     # (B*T, d_model, H*W)

        if self.positional_encoder is not None:
            pos_h = torch.arange(h, device=x.device).float().unsqueeze(0)   # (1, H)
            pos_w = torch.arange(w, device=x.device).float().unsqueeze(0)   # (1, W)
            pe_h = self.positional_encoder(pos_h)[0]                        # (H, d_model/2)
            pe_w = self.positional_encoder(pos_w)[0]                        # (W, d_model/2)
            pe = torch.cat(
                [pe_h[:, None, :].expand(h, w, -1),                         # riga
                 pe_w[None, :, :].expand(h, w, -1)],                        # colonna
                dim=-1,
            )                                                               # (H, W, d_model)
            pe = pe.view(h * w, self.d_model).t()                           # (d_model, H*W)
            out = out + pe.unsqueeze(0)

        # DIFF: query da tutti i pixel, key/value dalla mappa ridotta
        q_in = out.permute(0, 2, 1)                                         # (B*T, H*W, d_model)
        if self.sr is not None:
            kv_in = self.sr(out.view(-1, self.d_model, h, w)).flatten(2)    # (B*T, d_model, H*W/sr²)
            kv_in = kv_in.permute(0, 2, 1)                                  # (B*T, H*W/sr², d_model)
        else:
            kv_in = q_in

        out, attn = self.attention_heads(q_in, kv_in)
        # out: (n_head, B*T, H*W, d_model/n_head)

        out = (
            out.permute(1, 2, 0, 3).contiguous().view(sz_b * seq_t * h * w, -1)
        )  # Concatenate heads -> (B*T*H*W, d_model)
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, seq_t, h, w, -1).permute(0, 4, 1, 2, 3)        # (B, C, T, H, W)

        attn = attn.view(self.n_head, sz_b, seq_t, h * w, -1)              # head x b x t x query x key

        out = x + self.gamma * out                                          # DIFF: residuo

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadSpatialAttention(nn.Module):
    """Multi-Head Self-Attention module
    Come MultiHeadAttention di LTAE, ma con una query per ogni elemento calcolata dai dati
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        # DIFF: query calcolata dall'input (in LTAE: self.Q = nn.Parameter fisso)
        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductSpatialAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q_in, kv_in):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, len_q, _ = q_in.size()
        len_k = kv_in.size(1)

        # DIFF: una query per elemento -> (n*b) x lq x dk
        q = self.fc1_q(q_in).view(sz_b, len_q, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)

        k = self.fc1_k(kv_in).view(sz_b, len_k, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk

        v = torch.stack(kv_in.split(kv_in.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, len_k, -1
        )  # (n*b) x lk x (d_in/n)

        output, attn = self.attention(q, k, v)

        # DIFF: un output per query, niente squeeze
        attn = attn.view(n_head, sz_b, len_q, len_k)
        output = output.view(n_head, sz_b, len_q, d_in // n_head)

        return output, attn


class ScaledDotProductSpatialAttention(nn.Module):
    """Scaled Dot-Product Attention con più query
    Come ScaledDotProductAttention di LTAE, senza unsqueeze(1) sulla query
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(1, 2))     # DIFF: (n*b) x lq x lk invece di (n*b) x 1 x lk
        attn = attn / self.temperature
        attn = self.softmax(attn)                     # normalizza ogni riga (query) sulle key
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)                # (n*b) x lq x (d_in/n)
        return output, attn