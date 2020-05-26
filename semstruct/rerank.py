import argparse
import itertools
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
    parser.add_argument('-tmat', help='File containing transformation matrix.')
    parser.add_argument('-score_bleu', help='Calculate BLEU score wrt reference translation.')
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    if args.tmat:
        with open(args.tmat, 'rb') as f:
            tmat = torch.load(f, map_location='cpu')
    else:
        tmat = None

    with open(args.embeddings, 'rb') as f:
        indices, embeddings = torch.load(f)

    embsize = tmat.shape[1] if args.tmat else embeddings.shape[1]
    model = semstruct.PairwiseRanker(embsize, transformation_matrix=tmat).eval()

    with open(args.model, 'rb') as f:
        model.load_state_dict(torch.load(f))

    with open(args.sntlog, 'r') as f:
        sntlog = {k: [t for i, t in g] for k, g in itertools.groupby(scan_sntlog(f), lambda tup: tup[0])}

    output = []

    for i in torch.unique_consecutive(indices):
        embs = embeddings[indices == i]
        n = embs.shape[0]
        pairs = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j], dtype=torch.long)
        pred = model(embs[pairs[:, 0]], embs[pairs[:, 1]])
        scores = torch.zeros(n, n)
        scores[pairs[:, 0], pairs[:, 1]] = torch.sigmoid(pred).squeeze()
        # We run the classifier both ways, so the output may not be symmetric.
        sym_scores = torch.sqrt(scores * (1 - scores.t()))
        # remove diagonal
        compressed_sym = torch.tril(sym_scores)[:, :-1] + torch.triu(sym_scores)[:, 1:]
        hard_scores = compressed_sym > .5
        votes = torch.sum(hard_scores, dim=1)
        winner = torch.argmax(votes).item()
        output.append(sntlog[i.item()][winner])

    for snt in output:
        print(snt)

    if args.score_bleu:
        with open(args.score_bleu, 'r') as f:
            reference = [[line.rstrip('\n') for line in f]]

        print(sacrebleu.corpus_bleu(output, reference), file=sys.stderr)


if __name__ == '__main__':
    main()
