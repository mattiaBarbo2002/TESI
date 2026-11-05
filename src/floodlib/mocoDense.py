import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
https://github.com/WXinlong/DenseCL/

"""


class MoCo2encoders_dense(nn.Module):                                                  # MODIFICATO: nuovo nome


    def __init__(
        self,
        base_encoder_q,
        base_encoder_k,
        dim: int = 128,                 # dim vettori in uscita (globali e densi)
        K: int = 4096,                  # dimensione delle code
        m: float = 0.999,               # momentum
        T: float = 0.07,                # temperatura
        hidden_channels_dim: int = 64, # canali in uscita dalle conv della testa
        pool_grid: int = 4,             # griglia del pooling per il ramo globale
        dense_lambda: float = 0.5,      # peso della loss densa
        n_pos: int = 256,               # posizioni campionate per serie nella loss densa
        n_neg: int = 4,                 # NUOVO: posizioni per serie inserite nella coda densa
    ) -> None:

        super().__init__()                                                              # MODIFICATO: senza nome della classe

        self.K = K
        self.m = m
        self.T = T
        self.pool_grid = pool_grid
        self.dense_lambda = dense_lambda
        self.n_pos = n_pos
        self.n_neg = n_neg                                                              # NUOVO

        # encoder congelati
        self.encoder_q = base_encoder_q
        self.encoder_k = base_encoder_k

        # pesi encoder congelati, entrambi false, alleno solo MLP
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_q.requires_grad = False
            param_k.requires_grad = False  

        c_in_q = self.encoder_q.output_dim
        c_in_k = self.encoder_k.output_dim
        
        # testa convoluzionale (v5): 16 -> c/2 -> c, risoluzione 64x64 invariata
        self.conv_q = nn.Sequential(
            nn.Conv2d(c_in_q, hidden_channels_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels_dim//2),
            nn.ReLU(),

            nn.Conv2d(hidden_channels_dim//2, hidden_channels_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels_dim),
            nn.ReLU(),
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(c_in_k,hidden_channels_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels_dim//2),
            nn.ReLU(),

            nn.Conv2d(hidden_channels_dim//2, hidden_channels_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels_dim),
            nn.ReLU(),
        )

        # ramo globale: pooling + MLP -> un vettore per serie
        self.pool_q = nn.AdaptiveAvgPool2d(pool_grid)
        self.pool_k = nn.AdaptiveAvgPool2d(pool_grid)

        proj_in = hidden_channels_dim * pool_grid * pool_grid

        self.proj_q = nn.Sequential(
            nn.Linear(proj_in, dim), 
            nn.ReLU(), 
            nn.Linear(dim, dim)
        )

        self.proj_k = nn.Sequential(
            nn.Linear(proj_in, dim), 
            nn.ReLU(), 
            nn.Linear(dim, dim)
        )

        # ramo denso: MLP applicato a ogni posizione (conv 1x1) -> un vettore per posizione
        self.dense_q = nn.Sequential(
            nn.Conv2d(hidden_channels_dim, hidden_channels_dim, kernel_size=1, bias=False),  # MODIFICATO: bias=False
            nn.BatchNorm2d(hidden_channels_dim),                                             # NUOVO: strato nascosto normalizzato
            nn.ReLU(), 
            nn.Conv2d(hidden_channels_dim, dim, kernel_size=1, bias=False),                  # MODIFICATO: bias=False
            nn.BatchNorm2d(dim),                                                             # NUOVO: toglie la parte comune
        )

        self.dense_k = nn.Sequential(
            nn.Conv2d(hidden_channels_dim, hidden_channels_dim, kernel_size=1, bias=False),  # MODIFICATO: bias=False
            nn.BatchNorm2d(hidden_channels_dim),                                             # NUOVO
            nn.ReLU(), 
            nn.Conv2d(hidden_channels_dim, dim, kernel_size=1, bias=False),                  # MODIFICATO: bias=False
            nn.BatchNorm2d(dim),                                                             # NUOVO
        )

        # rami k: stessi pesi iniziali dei rami q, aggiornati solo con il momentum
        for param_q, param_k in zip(self.conv_q.parameters(), self.conv_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
 
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False    

        for param_q, param_k in zip(self.dense_q.parameters(), self.dense_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False                

        # coda globale: vettori globali SAR di altre serie
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)                    
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # coda densa: n_neg vettori di posizioni casuali per serie (mappa densa SAR)   
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)                     
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

    # peso(k) = peso(k) * m + peso(q) * (1 - m), per tutti e tre i rami
    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        for param_q, param_k in zip(self.conv_q.parameters(), self.conv_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(self.dense_q.parameters(), self.dense_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)          

    # coda circolare: sovrascrive i vettori più vecchi
    @torch.no_grad()
    def _enqueue(self, queue, ptr_buffer, keys) -> None:
        batch_size = keys.shape[0]
        ptr = int(ptr_buffer)
        assert self.K % batch_size == 0
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr_buffer[0] = (ptr + batch_size) % self.K

    # mescola il batch delle chiavi (per la batchnorm) e restituisce l'indice per riordinarlo
    @torch.no_grad()
    def _batch_shuffle(self, x):
        idx_shuffle = torch.randperm(x.shape[0], device=x.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    def contrastive_loss(self, im_q, im_k):

        # ramo ottico (q): gradiente solo sulla testa
        with torch.no_grad():
            z_q = self.encoder_q(im_q)

        f_q = self.conv_q(z_q)                                                          # (B, c, 64, 64)
        q = nn.functional.normalize(self.proj_q(self.pool_q(f_q).flatten(1)), dim=1)    # (B, dim)
        dq = nn.functional.normalize(self.dense_q(f_q), dim=1)                          # (B, dim, 64, 64)

        # ramo SAR (k): nessun gradiente
        with torch.no_grad():

            im_k_, idx_unshuffle = self._batch_shuffle(im_k)
            f_k = self.conv_k(self.encoder_k(im_k_))
            k = nn.functional.normalize(self.proj_k(self.pool_k(f_k).flatten(1)), dim=1)
            dk_map = self.dense_k(f_k)
            dk = nn.functional.normalize(dk_map, dim=1)                                  # (B, dim, 64, 64)

            k = k[idx_unshuffle]
            dk = dk[idx_unshuffle]

            # NUOVO: negativi locali, n_neg posizioni casuali per serie invece della media su tutta la mappa
            Bk, Dk, Hk, Wk = dk.shape
            sel = torch.randint(0, Hk * Wk, (Bk, self.n_neg), device=dk.device)             # (B, n_neg)
            dk_neg = torch.gather(dk.flatten(2), 2, sel.unsqueeze(1).expand(-1, Dk, -1))    # (B, dim, n_neg)
            dk_neg = dk_neg.permute(0, 2, 1).reshape(-1, Dk)                                 # (B*n_neg, dim)

        # loss globale: positivo = stessa serie, negativi = coda globale
        l_pos = (q * k).sum(dim=1, keepdim=True)                                         # (B, 1)
        l_neg = q @ self.queue.clone().detach()                                          # (B, K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss_g = nn.functional.cross_entropy(logits, labels)

        # loss densa: n_pos posizioni per serie, positivo = stessa posizione, negativi = coda densa
        B, D, H, W = dq.shape
        idx = torch.randint(0, H * W, (B, self.n_pos), device=dq.device)                 # (B, P)
        idx = idx.unsqueeze(1).expand(-1, D, -1)                                         # (B, D, P)
        dq_s = torch.gather(dq.flatten(2), 2, idx)                                       # (B, D, P)
        dk_s = torch.gather(dk.flatten(2), 2, idx)                                       # (B, D, P) stesse posizioni

        l_pos_d = (dq_s * dk_s).sum(dim=1).reshape(-1, 1)                                # (B*P, 1)
        dq_s = dq_s.permute(0, 2, 1).reshape(-1, D)                                      # (B*P, D)
        l_neg_d = dq_s @ self.queue2.clone().detach()                                    # (B*P, K)
        logits_d = torch.cat([l_pos_d, l_neg_d], dim=1) / self.T
        labels_d = torch.zeros(logits_d.shape[0], dtype=torch.long, device=logits_d.device)
        loss_d = nn.functional.cross_entropy(logits_d, labels_d)

        loss = (1 - self.dense_lambda) * loss_g + self.dense_lambda * loss_d
        return loss, loss_g, loss_d, k, dk_neg                                           

    def forward(self, im_q, im_k):
        # 1. aggiornamento dei rami k
        with torch.no_grad():
            self._momentum_update_key_encoder()

        # 2. loss
        loss, loss_g, loss_d, k, dk_neg = self.contrastive_loss(im_q, im_k)             

        # 3. nuove chiavi nelle code
        self._enqueue(self.queue, self.queue_ptr, k)
        self._enqueue(self.queue2, self.queue2_ptr, dk_neg)                             

        return loss, loss_g, loss_d

    # per l'anomaly detection: similarità SAR-ottico in ogni posizione
    @torch.no_grad()
    def similarity_map(self, im_q, im_k):
        dq = nn.functional.normalize(self.dense_q(self.conv_q(self.encoder_q(im_q))), dim=1)
        dk = nn.functional.normalize(self.dense_k(self.conv_k(self.encoder_k(im_k))), dim=1)
        return (dq * dk).sum(dim=1)         
















class MoCo2encoders_dense3(nn.Module):                                                   


    def __init__(
        self,
        base_encoder_q,
        base_encoder_k,
        dim: int = 128,                 # dim vettori in uscita (globali e densi)
        K: int = 4096,                  # dimensione delle code
        m: float = 0.999,               # momentum
        T: float = 0.07,                # temperatura della loss globale
        hidden_channels_dim: int = 64,  # canali in uscita dalle conv della testa
        pool_grid: int = 4,             # griglia del pooling per il ramo globale
        dense_lambda: float = 0.5,      # peso della loss densa (l'handler lo fa crescere da 0)
        n_pos: int = 64,                # celle usate per serie (64 = tutte, con griglia 8x8)
        n_neg: int = 4,                 # celle per serie inserite nella coda densa
        grid_dense: int = 8,            # mappa densa  ridotta a grid_dense x grid_dense celle
        T_dense: float = 0.2,           # T loss densa
    ) -> None:

        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.pool_grid = pool_grid
        self.dense_lambda = dense_lambda
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.grid_dense = grid_dense                                                     
        self.T_dense = T_dense                                                           

        # encoder congelati
        self.encoder_q = base_encoder_q
        self.encoder_k = base_encoder_k

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_q.requires_grad = False
            param_k.requires_grad = False

        c_in_q = self.encoder_q.output_dim
        c_in_k = self.encoder_k.output_dim

        # testa convoluzionale -> uguale net_v5
        self.conv_q = nn.Sequential(
            nn.Conv2d(c_in_q, hidden_channels_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels_dim//2),
            nn.ReLU(),

            nn.Conv2d(hidden_channels_dim//2, hidden_channels_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels_dim),
            nn.ReLU(),
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(c_in_k, hidden_channels_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels_dim//2),
            nn.ReLU(),

            nn.Conv2d(hidden_channels_dim//2, hidden_channels_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels_dim),
            nn.ReLU(),
        )

        # ramo globale -> uguale moco normale
        self.pool_q = nn.AdaptiveAvgPool2d(pool_grid)
        self.pool_k = nn.AdaptiveAvgPool2d(pool_grid)

        proj_in = hidden_channels_dim * pool_grid * pool_grid

        self.proj_q = nn.Sequential(
            nn.Linear(proj_in, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.proj_k = nn.Sequential(
            nn.Linear(proj_in, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # ramo denso  
        self.pool_dense = nn.AdaptiveAvgPool2d(grid_dense)                                     

        self.dense_q = nn.Sequential(
            nn.Conv2d(hidden_channels_dim, hidden_channels_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_channels_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.dense_k = nn.Sequential(
            nn.Conv2d(hidden_channels_dim, hidden_channels_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_channels_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        # rami k: stessi pesi iniziali dei rami q, aggiornati solo con il momentum
        for param_q, param_k in zip(self.conv_q.parameters(), self.conv_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.dense_q.parameters(), self.dense_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # coda globale: vettori globali SAR di altre serie
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # coda densa: n_neg celle rand per serie (da mappa SAR)
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

    # peso(k) = peso(k) * m + peso(q) * (1 - m) per tutti i rami
    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        for param_q, param_k in zip(self.conv_q.parameters(), self.conv_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(self.dense_q.parameters(), self.dense_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    # coda circolare: sovrascrive i vettori più vecchi
    @torch.no_grad()
    def _enqueue(self, queue, ptr_buffer, keys) -> None:
        batch_size = keys.shape[0]
        ptr = int(ptr_buffer)
        assert self.K % batch_size == 0
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr_buffer[0] = (ptr + batch_size) % self.K

    # mescola il batch delle chiavi (per la batchnorm), restituisce indice per riordino
    @torch.no_grad()
    def _batch_shuffle(self, x):
        idx_shuffle = torch.randperm(x.shape[0], device=x.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    def contrastive_loss(self, im_q, im_k):

        # ramo ottico (q): gradiente solo sulla testa
        with torch.no_grad():
            z_q = self.encoder_q(im_q)

        f_q = self.conv_q(z_q)                                                          # (B, c, 64, 64)
        q = nn.functional.normalize(self.proj_q(self.pool_q(f_q).flatten(1)), dim=1)    # (B, dim)
        dq = nn.functional.normalize(self.dense_q(self.pool_dense(f_q)), dim=1)        

        # ramo SAR (k): nessun gradiente
        with torch.no_grad():

            im_k_, idx_unshuffle = self._batch_shuffle(im_k)
            f_k = self.conv_k(self.encoder_k(im_k_))
            k = nn.functional.normalize(self.proj_k(self.pool_k(f_k).flatten(1)), dim=1)
            dk = nn.functional.normalize(self.dense_k(self.pool_dense(f_k)), dim=1)    

            k = k[idx_unshuffle]
            dk = dk[idx_unshuffle]

            # negativi locali: n_neg celle casuali per serie
            Bk, Dk, Hk, Wk = dk.shape
            sel = torch.randint(0, Hk * Wk, (Bk, self.n_neg), device=dk.device)
            dk_neg = torch.gather(dk.flatten(2), 2, sel.unsqueeze(1).expand(-1, Dk, -1))
            dk_neg = dk_neg.permute(0, 2, 1).reshape(-1, Dk)                             # (B*n_neg, dim)

        # loss globale: positivo = stessa serie, negativi = coda globale
        l_pos = (q * k).sum(dim=1, keepdim=True)
        l_neg = q @ self.queue.clone().detach()
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss_g = nn.functional.cross_entropy(logits, labels)

        # loss densa: positivo = stessa cella, negativi = coda densa
        B, D, H, W = dq.shape
        if self.n_pos >= H * W:                                                          
            dq_s = dq.flatten(2)                                                         # (B, D, H*W)
            dk_s = dk.flatten(2)
        else:
            idx = torch.randint(0, H * W, (B, self.n_pos), device=dq.device)
            idx = idx.unsqueeze(1).expand(-1, D, -1)
            dq_s = torch.gather(dq.flatten(2), 2, idx)
            dk_s = torch.gather(dk.flatten(2), 2, idx)

        l_pos_d = (dq_s * dk_s).sum(dim=1).reshape(-1, 1)                                # (B*P, 1)
        dq_s = dq_s.permute(0, 2, 1).reshape(-1, D)                                      # (B*P, D)
        l_neg_d = dq_s @ self.queue2.clone().detach()                                    # (B*P, K)
        logits_d = torch.cat([l_pos_d, l_neg_d], dim=1) / self.T_dense                   
        labels_d = torch.zeros(logits_d.shape[0], dtype=torch.long, device=logits_d.device)
        loss_d = nn.functional.cross_entropy(logits_d, labels_d)

        loss = (1 - self.dense_lambda) * loss_g + self.dense_lambda * loss_d
        return loss, loss_g, loss_d, k, dk_neg

    def forward(self, im_q, im_k):
        # aggiornamento dei rami k
        with torch.no_grad():
            self._momentum_update_key_encoder()

        # loss
        loss, loss_g, loss_d, k, dk_neg = self.contrastive_loss(im_q, im_k)

        # inserimento nuove chiavi nelle code
        self._enqueue(self.queue, self.queue_ptr, k)
        self._enqueue(self.queue2, self.queue2_ptr, dk_neg)

        return loss, loss_g, loss_d

    # per ad: similarità in ogni cella
    @torch.no_grad()
    def similarity_map(self, im_q, im_k):
        dq = nn.functional.normalize(self.dense_q(self.pool_dense(self.conv_q(self.encoder_q(im_q)))), dim=1)   
        dk = nn.functional.normalize(self.dense_k(self.pool_dense(self.conv_k(self.encoder_k(im_k)))), dim=1)  
        return (dq * dk).sum(dim=1)                                                                               # (B, 8, 8)    