# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        base_encoder,
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
        mlp: bool = False,
        symmetric: bool = False,
        device='cuda',
    ) -> None:
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(output_dim=dim, device=device)
        self.encoder_k = base_encoder(output_dim=dim, device=device)
        self.symmetric = symmetric

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        # idx_shuffle = torch.randperm(x.shape[0]).cuda()
        idx_shuffle = torch.randperm(x['im1'].shape[0]).to(device=device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        x['im1'] = x['im1'][idx_shuffle]
        x['im2'] = x['im2'][idx_shuffle]

        return x, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        """ contrastive loss (InfoNCE) computation """
        # compute query features
        # print("im_q.shape: ", im_q.shape)  # im_q.shape: torch.Size([batch-size, 1, 1280, 128])
        print("im_q.type: ", type(im_q))
        q = self.encoder_q(im_q)  # queries: NxC # N: batch size, C: moco-dim
        q = nn.functional.normalize(q, dim=1)  # already normalized
        print("contrastive_loss: q.shape:", q.shape)    # q.shape: torch.Size([batch-size, 128])

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        print("contrastive_loss: k.shape:", k.shape)  # k.shape: torch.Size([batch-size, 128])

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK  (K: dictionary size)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        del l_pos, l_neg

        # labels: positive key indicators
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
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # print("ModelMoCoDeeplab forward:", type(im1))
        # print("ModelMoCoDeeplab forward:", im1.shape)

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # symmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

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
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
        symmetric: bool = False,
        device='cuda',
    ) -> None:
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo2encoders, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder_q
        self.encoder_k = base_encoder_k
        self.symmetric = symmetric

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_q.requires_grad = False
            param_k.requires_grad = False  # not update by gradient

        encoder_q_dim = self.encoder_q.fc.weight.shape[0]#*self.encoder_q.weight.shape[2]*self.encoder_q.weight.shape[3]*self.encoder_q.weight.shape[4]
        self.proj_q = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(encoder_q_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        encoder_k_dim = self.encoder_k.fc.weight.shape[0]#*self.encoder_k.weight.shape[2]*self.encoder_k.weight.shape[3]*self.encoder_k.weight.shape[4]
        self.proj_k = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(encoder_k_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        for param_proj_q, param_proj_k in zip(
            self.proj_q.parameters(), self.proj_k.parameters()
        ):
            param_proj_k.data.copy_(param_proj_q.data)  # initialize
            param_proj_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.proj_q.parameters(), self.proj_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        # idx_shuffle = torch.randperm(x.shape[0]).cuda()
        # idx_shuffle = torch.randperm(x['im1'].shape[0]).to(device=device)
        idx_shuffle = torch.randperm(x.shape[0]).to(device=device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        x = x[idx_shuffle]
        # x['im2'] = x['im2'][idx_shuffle]

        return x, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        """ contrastive loss (InfoNCE) computation """
        # compute query features
        # print("im_q.shape: ", im_q.shape)  # im_q.shape: torch.Size([batch-size, 1, 1280, 128])
        print("im_q.type: ", type(im_q))
        with torch.no_grad():
            q = self.encoder_q(im_q)  # queries: NxC # N: batch size, C: moco-dim
        q = nn.functional.normalize(self.proj_q(q), dim=1)  # already normalized
        print("contrastive_loss: q.shape:", q.shape)    # q.shape: torch.Size([batch-size, 128])

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(self.proj_k(k), dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        print("contrastive_loss: k.shape:", k.shape)  # k.shape: torch.Size([batch-size, 128])

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK  (K: dictionary size)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        del l_pos, l_neg

        # labels: positive key indicators
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
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # print("ModelMoCoDeeplab forward:", type(im1))
        # print("ModelMoCoDeeplab forward:", im1.shape)

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # symmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        print("ModelMoCoUnet(nn.Module)", "minibatch loss value", loss)

        return loss

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
