import argparse
import itertools
import logging
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_set', help='Sentence embeddings for training.')
    parser.add_argument('validation_set', help='Sentence embeddings for validation.')
    parser.add_argument('outfile', help='Output file for trained matrix.')
    parser.add_argument('-checkpoint', help='File name to save checkpoints after each epoch.')
    parser.add_argument('-dims', type=int, default=10,
                        help='Number of dimensions for discriminative part of embeddings.')
    parser.add_argument('-poolsize', type=int, default=20,
                        help='Number of sentences to mix examples from.')
    parser.add_argument('-batchsize', type=int, default=20,
                        help='Batchsize per weight update.')
    parser.add_argument('-epochs', type=int, default=3, help='Number of epochs to train.')
    parser.add_argument('-lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('-weight-by-dims', action='store_true',
                        help='Weight loss function contribution of similarity and discrimination inversely ' +
                        'by number of dimension.')
    parser.add_argument('-device', default='cpu', help='CUDA device to use, if any.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

    device = torch.device(args.device)

    with open(args.training_set, 'rb') as f:
        training_set = torch.load(f)
        training_set[1] = training_set[1].to(device)

    with open(args.validation_set, 'rb') as f:
        validation_set = torch.load(f)
        validation_set[1] = validation_set[1].to(device)

    tmat = train(training_set, validation_set, args, device=device)

    with open(args.outfile, 'wb') as f:
        torch.save(tmat, f)


def make_batches(iterable, batchsize):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batchsize))
        if len(batch) == 0:
            return
        yield batch


def make_pair_batches(indices, poolsize, batchsize, device='cpu'):
    offset = 0
    for pool in make_batches([sum(1 for _ in g) for k, g in itertools.groupby(indices)], poolsize):
        pool_t = torch.tensor(pool, dtype=torch.long)
        npairs = torch.sum(pool_t * (pool_t - 1)).item()
        pool_pairs = torch.empty(npairs, 2, dtype=torch.long, device=device)
        start = 0
        for n in pool:
            for i in range(n):
                pool_pairs[(start + i * (n - 1)):(start + (i + 1) * (n - 1)), 0] = offset + i
                js = torch.tensor([j for j in range(n) if i != j], dtype=torch.long, device=device)
                pool_pairs[(start + i * (n - 1)):(start + (i + 1) * (n - 1)), 1] = offset + js
            offset += n
            start += n * (n - 1)
        perm = torch.randperm(npairs, device=device)
        for i in range(0, npairs, batchsize):
            yield pool_pairs[perm[i:(i + batchsize)]]


def make_optimiser(args, params):
    opt = torch.optim.Adam(params)
    return opt


def train(training_set, validation_set, args, device='cpu'):
    indices, embeddings = training_set

    embsize = embeddings.shape[1]
    paramsize = embsize * (embsize + 1) // 2

    matc = TransformationMatrix(embsize, args.dims, weight_by_dims=args.weight_by_dims, device=device)

    param = 2 * torch.rand(paramsize, device=device) - 1
    param.requires_grad_()

    opt = make_optimiser(args, [param])

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        logging.info('EPOCH %d' % epoch)
        for pairs in make_pair_batches(indices, args.poolsize, args.batchsize, device=device):
            opt.zero_grad()
            tmat = matc.transformation_matrix(param)
            loss = matc.compute_loss(tmat, embeddings, pairs)
            loss.backward()
            opt.step()

        with torch.no_grad():
            tmat = matc.transformation_matrix(param)
            train_loss, train_sim_loss, train_disc_loss = matc.score_set(tmat, training_set, args)
            logging.info('Epoch %d. Training loss: %g - Similarity loss: %g - Discrimination loss: %g' %
                         (epoch, train_loss, train_sim_loss, train_disc_loss))
            val_loss, val_sim_loss, val_disc_loss = matc.score_set(tmat, validation_set, args)
            logging.info('Epoch %d. Validation loss: %g - Similarity loss: %g - Discrimination loss: %g' %
                         (epoch, val_loss, val_sim_loss, val_disc_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                best_param = param.clone().detach()
                logging.info('New best validation loss: %g' % best_val_loss)

        if args.checkpoint:
            with open(args.checkpoint, 'wb') as f:
                torch.save(param.to('cpu'), f)

    return matc.transformation_matrix(best_param)


class TransformationMatrix:
    def __init__(self, n, dims, weight_by_dims=False, device='cpu'):
        self.device = device
        self.n = n
        idx = torch.tril_indices(n, n)
        self.idx0 = idx[0]
        self.idx1 = idx[1]
        self.dims = dims

        if weight_by_dims:
            self.disc_weight = 1 / dims
            self.sim_weight = 1 / (n - dims)
        else:
            self.disc_weight = 1.0
            self.sim_weight = 1.0

    def transformation_matrix(self, param):
        tri = torch.zeros(self.n, self.n, device=param.device)
        tri[self.idx0, self.idx1] = param
        ss = tri - tri.t()
        i_n = torch.eye(self.n, device=param.device)
        q = torch.solve(i_n + ss, i_n - ss).solution
        return q

    def compute_loss_components(self, tmat, embeddings, pairs):
        diff = embeddings[pairs[:, 0], :] - embeddings[pairs[:, 1], :]
        transformed = diff @ tmat
        sim_loss = torch.norm(transformed[:, self.dims:])
        disc_loss = -torch.norm(transformed[:, :self.dims])
        return sim_loss, disc_loss

    def compute_loss(self, tmat, embeddings, pairs):
        sim_loss, disc_loss = self.compute_loss_components(tmat, embeddings, pairs)
        return self.sim_weight * sim_loss + self.disc_weight * disc_loss

    def score_set(self, tmat, dataset, args):
        indices, embeddings = dataset
        total_loss = 0
        total_sim_loss = 0
        total_disc_loss = 0
        for pairs in make_pair_batches(indices, args.poolsize, args.batchsize, device=embeddings.device):
            sim_loss, disc_loss = self.compute_loss_components(tmat, embeddings, pairs)
            loss = self.sim_weight * sim_loss + self.disc_weight * disc_loss
            total_sim_loss += sim_loss
            total_disc_loss += disc_loss
            total_loss += loss
        return total_loss, total_sim_loss, total_disc_loss


if __name__ == '__main__':
    main()
