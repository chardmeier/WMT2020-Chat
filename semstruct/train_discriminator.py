import argparse
import itertools
import random
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings', help='Sentence embeddings.')
    parser.add_argument('outfile', help='Output file for trained matrix.')
    parser.add_argument('-dims', type=int, default=10,
                        help='Number of dimensions for discriminative part of embeddings.')
    parser.add_argument('-poolsize', type=int, default=20,
                        help='Number of sentences to mix examples from.')
    parser.add_argument('-batchsize', type=int, default=20,
                        help='Batchsize per weight update.')
    parser.add_argument('-epochs', type=int, default=3, help='Number of epochs to train.')
    parser.add_argument('-lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('-momentum', type=float, default=0.9, help='Momentum.')
    args = parser.parse_args()

    with open(args.embeddings, 'rb') as f:
        indices, embeddings = torch.load(f)

    tmat = train(indices, embeddings, args)

    with open(args.outfile, 'wb') as f:
        torch.save(tmat, f)


class TransformationMatrixCreator:
    def __init__(self, n):
        self.n = n
        idx = torch.tril_indices(n, n)
        self.idx0 = idx[0]
        self.idx1 = idx[1]

    def tmat(self, param):
        tri = torch.zeros(self.n, self.n)
        tri[self.idx0, self.idx1] = param
        q, r = torch.qr(tri)
        return q


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


def train(indices, embeddings, args):
    embsize = embeddings.shape[1]
    paramsize = embsize * (embsize + 1) // 2

    matc = TransformationMatrixCreator(embsize)

    param = 2 * torch.rand(paramsize) - 1
    param.requires_grad = True

    opt = make_optimiser(args, [param])

    for epoch in range(args.epochs):
        for pairs in make_batches(make_pairs(indices, args.poolsize), args.batchsize):
            opt.zero_grad()
            tmat = matc.tmat(param)
            pairs_t = torch.LongTensor(pairs)
            diff = embeddings[pairs_t[:, 0], :] - embeddings[pairs_t[:, 1], :]
            proj = tmat @ diff
            loss = torch.norm(proj[:, args.dims:]) - torch.norm(proj[:, :args.dims])
            loss.backward()
            opt.step()

    return matc.tmat(param)


if __name__ == '__main__':
    main()
