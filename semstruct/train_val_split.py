import argparse
import itertools
import random
import sys
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_out', required=True, help='Filename for training set.')
    parser.add_argument('-val_out', required=True, help='Filename for validation set.')
    parser.add_argument('-val_indices', help='File containing indices of validation sentences.')
    parser.add_argument('-val_size', type=int, help='Number of sentences to include in validation set.')
    parser.add_argument('inputs', nargs='+', help='Input files.')
    args = parser.parse_args()

    if not args.val_indices and not args.val_size:
        print('One of -val_indices or -val_size must be specified.', file=sys.stderr)
        sys.exit(1)

    sentences = {}
    for inpfile in args.inputs:
        print('Reading %s.' % inpfile, file=sys.stderr)
        with open(inpfile, 'rb') as f:
            indices, embeddings = torch.load(f)
        for i in torch.unique_consecutive(indices):
            embs = embeddings[indices == i, :]
            if i in sentences:
                sentences[i] = torch.cat([sentences[i], embs], dim=0)
            else:
                sentences[i] = embs

    if args.val_indices:
        print('Reading validation indices from %s.' % args.val_indices, file=sys.stderr)
        with open(args.val_indices, 'r') as f:
            val_indices = [int(line.rstrip('\n')) for line in f]
    else:
        nsent = indices[-1] + 1
        print('Found %d sentences. Generating random sample of size %d.' % (nsent, args.val_size), file=sys.stderr)
        val_indices = random.sample(sentences.keys(), args.val_size)
        val_indices.sort()
        for i in val_indices:
            print(i)

    train_indices = list(sorted(set(sentences.keys()) - set(val_indices)))

    print('Writing training set to %s.' % args.train_out, file=sys.stderr)
    create_set(sentences, args.train_out, train_indices)

    print('Writing validation set to %s.' % args.val_out, file=sys.stderr)
    create_set(sentences, args.val_out, val_indices)


def create_set(sentences, outfile, indices):
    embs = torch.cat([sentences[i] for i in indices])
    inds = []
    for i in indices:
        inds.extend(i for _ in range(sentences[i].shape[0]))
    inds = torch.LongTensor(inds)
    with open(outfile, 'wb') as f:
        torch.save([inds, embs], f)


if __name__ == '__main__':
    main()
