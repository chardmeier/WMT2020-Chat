import argparse
import itertools
import math
import sacrebleu
import semstruct
import sys
import torch


def scan_sntlog(f):
    for line in f:
        if ' ||| ' in line:
            fields = line.rstrip('\n').split(' ||| ')
            yield int(fields[0]), fields[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Reranking model to use.')
    parser.add_argument('embeddings', help='N-best embeddings to rerank.')
    parser.add_argument('sntlog', help='Sentences corresponding to n-best embeddings.')
    parser.add_argument('-score_bleu', help='Calculate BLEU score wrt reference translation.')
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = torch.load(f)

    with open(args.embeddings, 'rb') as f:
        indices, embeddings = torch.load(f)

    with open(args.sntlog, 'r') as f:
        sntlog = {k: [t for i, t in g] for k, g in itertools.groupby(scan_sntlog(f), lambda tup: tup[0])}

    output = []

    threshold = math.log(.5)
    for i in sorted(sntlog.keys()):
        embs = embeddings[indices == i]
        n = embs.shape[0]
        pairs = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j], dtype=torch.long)
        pred = model(embs[pairs[:, 0]], embs[pairs[:, 1]])
        scores = torch.zeros(n, n)
        scores[pairs[:, 0], pairs[:, 1]] = torch.nn.functional.logsigmoid(pred)
        # We run the classifier both ways, so the output may not be symmetric.
        hard_scores = ((scores + scores.t()) / 2) > threshold
        votes = torch.sum(hard_scores, dim=1)
        winner = torch.argmax(votes)
        output.append(sntlog[i][winner])

    for snt in output:
        print(snt)

    if args.score_bleu:
        with open(args.score_bleu, 'r') as f:
            reference = [[line.rstrip('\n') for line in f]]

        print(sacrebleu.corpus_bleu(output, reference), file=sys.stderr)


if __name__ == '__main__':
    main()
