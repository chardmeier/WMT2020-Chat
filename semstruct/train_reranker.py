import argparse
import itertools
import logging
import random
import sys
import torch
import tqdm


class PairwiseRanker(torch.nn.Module):
    def __init__(self, embsize, transformation_matrix=None):
        super().__init__()
        self.embsize = embsize
        self.linear = torch.nn.Linear(2 * embsize, 1)
        self.transformation_matrix = transformation_matrix

    def forward(self, x1, x2):
        if self.transformation_matrix is not None:
            tx1 = x1 @ self.transformation_matrix
            tx2 = x2 @ self.transformation_matrix
        else:
            tx1 = x1
            tx2 = x2
        return self.linear(torch.cat([tx1, tx2], dim=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_input', help='File containing input embeddings.')
    parser.add_argument('output', help='File to store trained weights in.')
    parser.add_argument('-train_scored_nbest', required=True, help='Scored n-best list for training set.')
    parser.add_argument('-val_scored_nbest', help='Scored n-best list for validation set.')
    parser.add_argument('-val', help='Input embeddings validation set.')
    parser.add_argument('-tmat', help='File containing transformation matrix.')
    parser.add_argument('-batchsize', type=int, default=50, help='Batch size for training.')
    parser.add_argument('-epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('-device', default='cpu', help='CUDA device to use, if any.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

    device = torch.device(args.device)

    with open(args.train_input, 'rb') as f:
        indices, embeddings = torch.load(f)
    with open(args.train_scored_nbest, 'r') as f:
        pairwise = load_scored_nbest(f)

    embeddings = embeddings.to(device)

    if args.val:
        if not args.val_scored_nbest:
            logging.error('-val_scored_nbest is required if using validation set.')
            sys.exit(1)

        with open(args.val, 'rb') as f:
            val_indices, val_embeddings = torch.load(f)
        with open(args.val_scored_nbest, 'r') as f:
            val_pairwise = load_scored_nbest(f)

    if args.tmat:
        with open(args.tmat, 'rb') as f:
            tmat = torch.load(f)
        embsize = tmat.shape[1]
    else:
        tmat = None
        embsize = embeddings.shape[1]

    model = PairwiseRanker(embsize, transformation_matrix=tmat).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

    opt = torch.optim.LBFGS(model.parameters())

    batches_per_epoch = len(pairwise) // args.batchsize

    for epoch in range(args.epochs):
        logging.info('EPOCH %d' % epoch)
        for x1, x2, y in tqdm.tqdm(make_examples(embeddings, pairwise, args.batchsize), total=batches_per_epoch):
            def train_closure():
                opt.zero_grad()
                y_hat = model(x1, x2)
                loss = loss_fn(y_hat, y)
                loss.backward()
                return loss
            opt.step(train_closure)

        with torch.no_grad():
            val_loss = 0
            for x1, x2, y in make_examples(val_embeddings, val_pairwise, args.batchsize):
                y_hat = model(x1, x2)
                val_loss += loss_fn(y_hat, y).item()
            logging.info('Validation loss: %g' % val_loss)

    with open(args.output, 'wb') as f:
        torch.save(model.to('cpu').state_dict(), f)


def scan_scored_nbest(input_file):
    for i, line in enumerate(input_file):
        fields = line.rstrip('\n').split(' ||| ')
        yield int(fields[0]), i, float(fields[2])


def load_scored_nbest(input_file):
    pairwise = []
    for k, g in itertools.groupby(scan_scored_nbest(input_file), lambda tup: tup[0]):
        group = [(i, score) for _, i, score in g]
        for i, score1 in group:
            for j, score2 in group:
                if score1 > score2:
                    pairwise.append((i, j, 1))
                    pairwise.append((j, i, 0))
                elif score1 < score2:
                    pairwise.append((i, j, 0))
                    pairwise.append((j, i, 1))
    return pairwise


def make_examples(embeddings, pairwise, batchsize):
    shuffled = random.sample(pairwise, k=len(pairwise))
    while shuffled:
        batch = shuffled[:batchsize]
        shuffled = shuffled[batchsize:]
        batch_t = torch.tensor(batch, dtype=torch.long, device=embeddings.device)
        x1 = embeddings[batch_t[:, 0]]
        x2 = embeddings[batch_t[:, 1]]
        y = batch_t[:, 2].unsqueeze(1).float()
        yield x1, x2, y


if __name__ == '__main__':
    main()
