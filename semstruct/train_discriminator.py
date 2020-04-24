import argparse
import itertools
import logging
import random
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
    parser.add_argument('-momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('-weight-by-dims', action='store_true',
                        help='Weight loss function contribution of similarity and discrimination inversely ' +
                        'by number of dimension.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

    with open(args.training_set, 'rb') as f:
        training_set = torch.load(f)

    with open(args.validation_set, 'rb') as f:
        validation_set = torch.load(f)

    tmat = train(training_set, validation_set, args)

    with open(args.outfile, 'wb') as f:
        torch.save(tmat, f)


def make_batches(iterable, batchsize):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batchsize))
        if len(batch) == 0:
            return
        yield batch


def make_pairs(indices, poolsize):
    offset = 0
    for pool in make_batches([sum(1 for _ in g) for k, g in itertools.groupby(indices)], poolsize):
        pool_pairs = []
        for n in pool:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        pool_pairs.append((offset + i, offset + j))
            offset += n
        random.shuffle(pool_pairs)
        yield from pool_pairs


def make_optimiser(args, params):
    opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    return opt


def train(training_set, validation_set, args):
    indices, embeddings = training_set

    embsize = embeddings.shape[1]
    paramsize = embsize * (embsize + 1) // 2

    matc = TransformationMatrix(embsize, args.dims, weight_by_dims=args.weight_by_dims)

    param = 2 * torch.rand(paramsize) - 1
    param.requires_grad_()

    opt = make_optimiser(args, [param])

    for epoch in range(args.epochs):
        logging.info('EPOCH %d' % epoch)
        for pairs in make_batches(make_pairs(indices, args.poolsize), args.batchsize):
            opt.zero_grad()
            loss = matc.compute_loss(param, embeddings, pairs)
            loss.backward()
            opt.step()

        with torch.no_grad():
            tri = matc.param_to_triangular(param)
            tmat = matc.triangular_to_tmat(tri)
            train_loss, train_sim_loss, train_disc_loss = matc.score_set(tmat, training_set, args)
            logging.info('Epoch %d. Training loss: %g - Similarity loss: %g - Discrimination loss: %g' %
                         (epoch, train_loss, train_sim_loss, train_disc_loss))
            val_loss, val_sim_loss, val_disc_loss = matc.score_set(tmat, validation_set, args)
            logging.info('Epoch %d. Validation loss: %g - Similarity loss: %g - Discrimination loss: %g' %
                         (epoch, val_loss, val_sim_loss, val_disc_loss))
            det = torch.prod(torch.diag(tri))
            logging.info('Epoch %d. Determinant of parameter matrix: %g' % (epoch, det))

        if args.checkpoint:
            with open(args.checkpoint, 'wb') as f:
                torch.save(param, f)

    return matc.transformation_matrix(param.detach())


class TransformationMatrix:
    def __init__(self, n, dims, weight_by_dims=False):
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

    def param_to_triangular(self, param):
        tri = torch.zeros(self.n, self.n)
        tri[self.idx0, self.idx1] = param
        return tri

    def triangular_to_tmat(self, tri):
        q, r = torch.qr(tri)
        return q

    def transformation_matrix(self, param):
        tri = self.param_to_triangular(param)
        q = self.triangular_to_tmat(tri)
        return q

    def compute_loss_components(self, tmat, embeddings, pairs):
        pairs_t = torch.LongTensor(pairs)
        diff = embeddings[pairs_t[:, 0], :] - embeddings[pairs_t[:, 1], :]
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
        for pairs in make_batches(make_pairs(indices, args.poolsize), args.batchsize):
            sim_loss, disc_loss = self.compute_loss_components(tmat, embeddings, pairs)
            loss = self.sim_weight * sim_loss + self.disc_weight * disc_loss
            total_sim_loss += sim_loss
            total_disc_loss += disc_loss
            total_loss += loss
        return total_loss, total_sim_loss, total_disc_loss


if __name__ == '__main__':
    main()
