# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# COMPORTAMENTO MoCo
# 1. encoder_k calcola i suoi pesi partendo da encoder_q -> peso(k) = peso(k)*m + peso(q)*(1-m)
# 2. encoder_q prende i dim_batch tensori ottici (im_q) e ritorna matrice di vettori [dim_batch x dim]
# 3. encoder_k prende i dim_batch tensori sar (im_k) e ritorna matrice di vettori [dim_batch x dim]
# 4. infoNCE tra matrice_q e coda
# 5. sovrascrivo gli ultimi dim_batch vettori di coda con matrice_k
# 6. modifica pesi encoder_q

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# serve per addestrare la rete da 0, io invece uso già gli encoder pre-allenati. Vedi MoCo2Encoders
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        base_encoder,
        dim: int = 128,             # dim vettore output
        K: int = 65536,             # queue size, k negativi
        m: float = 0.999,           # momentum
        T: float = 0.07,            # softmax temperature
        mlp: bool = False,
        symmetric: bool = False,
        device='cuda',
    ) -> None:
        
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # crea encoder q (queue_encoder) e k (key_encoder)
        self.encoder_q = base_encoder(output_dim=dim, device=device)
        self.encoder_k = base_encoder(output_dim=dim, device=device)
        self.symmetric = symmetric

        # entra nel singleEncoder e sostituisce il layer finale fully connected con piccola rete mlp
        # sia per econder q che k
        if mlp:  
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        # k copia i parametri di q, e upgrade k congelati
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  
            param_k.requires_grad = False  

        # crea coda
        # register_buffer non è un parametro da allenare, matrice random 128*K
        # normalizzazione vettori matrice a lunghezza 1
        # queue_ptr = indice coda (init 0)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # @no_grad: blocco che non calcola i gradienti, risparmio memoria
    # peso(k) = peso(k)*m + peso(q)*(1-m)
    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    # aggiornamento coda con nuovi valori calcolati da momentum_encoder
    # la matrice dim*K non viene mai cancellata ma sovrascritta
    # se batch = 32 sovrascrivo i 32 vettori più vecchi della matrice
    # si sovrascrive da dove dice ptr
    # ptr = (ptr + batch_size) % self.K se la matrice finisce ptr torna a 0 -> coda circolare
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
        
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  
    
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  

        self.queue_ptr[0] = ptr

    # disordino ordine chiavi k per non far barare batch_normalization
    # su una sola gpu non serve, di solito distribuisco il batch k su gpu diverse rispetto alla distribuzione del batch q
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        
        # idx_shuffle = torch.randperm(x.shape[0]).cuda()
        idx_shuffle = torch.randperm(x['im1'].shape[0]).to(device=device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        x['im1'] = x['im1'][idx_shuffle]
        x['im2'] = x['im2'][idx_shuffle]

        return x, idx_unshuffle

    # riordine vettori, altrimenti non so confrontare coppie positive
    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        
        return x[idx_unshuffle]

    # apprendimento
    # im_q, im_k = tensori 5D
    # q = matrice [dim_batch x dim (128)] di vettori del batch im_q processati da q_encoder
    def contrastive_loss(self, im_q, im_k):
        
        # print("im_q.shape: ", im_q.shape)  # im_q.shape: torch.Size([batch-size, 1, 1280, 128])
        print("im_q.type: ", type(im_q))
        q = self.encoder_q(im_q)  
        q = nn.functional.normalize(q, dim=1)  
        print("contrastive_loss: q.shape:", q.shape)   

        # encoder_k mescola chiavi, processa batch, riordina chiavi 
        # no gradient
        with torch.no_grad():  
        
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k_)  
            k = nn.functional.normalize(k, dim=1)  

            # riordina
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        print("contrastive_loss: k.shape:", k.shape)  

        # similarità positiva: moltiplica matrice q per colonna corretta k
        # vettori normalizzati quindi prodotto [-1,1]
        # output colonna dim_batch*1, ogni riga è il singolo punteggio della serie
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # similarità negativa: moltiplica la matrice q per la coda
        # output matrice dim_batch*dim_coda
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # aggiunge colonna positiva all'inizio della matrice negativa
        # output matrice dim_batch*(dim_coda+1)
        # divide per temperatura softmax, differenze minime esplodono
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        del l_pos, l_neg

        # labels = lista di 0 di dim_batch, indica che per ogni serie nel batch la coppia positiva è la prima colonna
        # si calcola loss tra matrice e vettore labels
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device=device)
        loss = nn.CrossEntropyLoss().to(device=device)(logits, labels)
        del logits, labels

        """
        from moco-paper: 
            'we encode the queries and their corresponding keys, which form the positive sample pairs. 
            The negative samples are from the queue'
        """
        print("contrastive_loss()", "minibatch loss value", loss)

        return loss, q, k

    def forward(self, im1, im2):
        # forward -> scorrono i dati in avanti, no backpropagation qui
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        
        # 1. aggiornamento pesi encoder_k
        with torch.no_grad():  
            self._momentum_update_key_encoder()

        # 2. contrastive loss: im_q e im_k passati agli encoder    
        # caso simmetrico: somma loss sar->ottico e ottico->sar
        if self.symmetric:  
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)

        # caso standard loss sar->ottico
        else:  
            loss, q, k = self.contrastive_loss(im1, im2)

        # 3. inserimento nuovi vettori in coda
        self._dequeue_and_enqueue(k)

        print("ModelMoCoUnet(nn.Module)", "minibatch loss value", loss)

        return loss


class MoCo2encoders(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        base_encoder_q,
        base_encoder_k,
        dim: int = 128,             # dim vettore output
        K: int = 65536,             # queue size, k negativi
        m: float = 0.999,           # momentum
        T: float = 0.07,            # softmax temperature
        symmetric: bool = False,
        device='cuda',
    ) -> None:
        
        super(MoCo2encoders, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # due encoder diversi, non più uguali
        self.encoder_q = base_encoder_q             # self.encoder_q = base_encoder(output_dim=dim, device=device)
        self.encoder_k = base_encoder_k             # self.encoder_k = base_encoder(output_dim=dim, device=device)
        self.symmetric = symmetric

        # pesi encoder congelati, entrambi false, alleno solo MLP
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_q.requires_grad = False
            param_k.requires_grad = False  

        # sostituzione layer fully connected con MLP per entrambi
        encoder_q_dim = self.encoder_q.fc.weight.shape[0]
        self.proj_q = nn.Sequential(
            nn.Linear(encoder_q_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        encoder_k_dim = self.encoder_k.fc.weight.shape[0]
        self.proj_k = nn.Sequential(
            nn.Linear(encoder_k_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )


        # inizializzazione MLP q (predefinita pytorch), e copia pesi per k
        for param_proj_q, param_proj_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_proj_k.data.copy_(param_proj_q.data)  # initialize
            param_proj_k.requires_grad = False  # not update by gradient

        # coda e indice
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # @no_grad: blocco che non calcola i gradienti, risparmio memoria
    # peso(k) = peso(k)*m + peso(q)*(1-m)
    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.proj_q.parameters(), self.proj_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    # aggiornamento coda con nuovi valori calcolati da momentum_encoder
    # la matrice dim*K non viene mai cancellata ma sovrascritta
    # se batch = 32 sovrascrivo i 32 vettori più vecchi della matrice
    # si sovrascrive da dove dice ptr
    # ptr = (ptr + batch_size) % self.K se la matrice finisce ptr torna a 0 -> coda circolare
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
        
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0 

        self.queue[:, ptr : ptr + batch_size] = keys.T
        # move pointer
        ptr = (ptr + batch_size) % self.K  

        self.queue_ptr[0] = ptr

    # disordino ordine chiavi k per non far barare batch_normalization
    # l'estrazione del dizionario viene gestito in loader.py in get_item
    # su una sola gpu non serve, di solito distribuisco il batch k su gpu diverse rispetto alla distribuzione del batch q
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
    
        # idx_shuffle = torch.randperm(x.shape[0]).cuda()
        # idx_shuffle = torch.randperm(x['im1'].shape[0]).to(device=device)
        idx_shuffle = torch.randperm(x.shape[0]).to(device=device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        x = x[idx_shuffle]
        # x['im2'] = x['im2'][idx_shuffle]

        return x, idx_unshuffle

    # riordine vettori, altrimenti non so confrontare coppie positive
    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        
        return x[idx_unshuffle]

    # apprendimento
    # im_q, im_k = tensori 5D
    # q = matrice [dim_batch x dim (128)] di vettori del batch im_q processati da q_encoder
    def contrastive_loss(self, im_q, im_k):
        
        print("im_q.type: ", type(im_q))
        # blocco no.grad non salvo valori per backpropagation
        with torch.no_grad():
            q = self.encoder_q(im_q)  
        q = nn.functional.normalize(self.proj_q(q), dim=1)  
        print("contrastive_loss: q.shape:", q.shape)    

        # encoder_k mescola chiavi, processa batch, riordina chiavi 
        # no gradient
        with torch.no_grad():  
            
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k_)  
            k = nn.functional.normalize(self.proj_k(k), dim=1) 

            # riordina
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        print("contrastive_loss: k.shape:", k.shape)  

        # similarità positiva: moltiplica matrice q per colonna corretta k
        # vettori normalizzati quindi prodotto [-1,1]
        # output colonna dim_batch*1, ogni riga è il singolo punteggio della serie
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # similarità negativa: moltiplica la matrice q per la coda
        # output matrice dim_batch*dim_coda
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # aggiunge colonna positiva all'inizio della matrice negativa
        # output matrice dim_batch*(dim_coda+1)
        # divide per temperatura softmax, differenze minime esplodono
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        del l_pos, l_neg

        # labels = lista di 0 di dim_batch, indica che per ogni serie nel batch la coppia positiva è la prima colonna
        # si calcola loss tra matrice e vettore labels
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device=device)
        loss = nn.CrossEntropyLoss().to(device=device)(logits, labels)
        del logits, labels

        """
        fromn moco-paper: 
            'we encode the queries and their corresponding keys, which form the positive sample pairs. 
            The negative samples are from the queue'
        """
        print("contrastive_loss()", "minibatch loss value", loss)

        return loss, q, k

    def forward(self, im1, im2):
        # forward -> scorrono i dati in avanti, no backpropagation qui
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # print("ModelMoCoDeeplab forward:", type(im1))
        # print("ModelMoCoDeeplab forward:", im1.shape)

        # 1. aggiornamento pesi encoder_k
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # 2. contrastive loss: im_q e im_k passati agli encoder    
        # caso simmetrico: somma loss sar->ottico e ottico->sar
        if self.symmetric:  
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)

        # caso standard loss sar->ottico
        else:  
            loss, q, k = self.contrastive_loss(im1, im2)

        # 3. inserimento nuovi vettori in coda
        self._dequeue_and_enqueue(k)

        print("ModelMoCoUnet(nn.Module)", "minibatch loss value", loss)

        return loss


# utils, probabilemente estratta da paper
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

