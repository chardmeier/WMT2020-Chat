import argparse
import sacrebleu
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', help='Reference translation.')
    parser.add_argument('nbest', help='N-best list to score.')
    parser.add_argument('-val_indices', help='File containing indices of validation sentences.')
    args = parser.parse_args()

    with open(args.reference, 'r') as f:
        refs = [line.rstrip('\n') for line in f]

    if args.val_indices:
        with open(args.val_indices, 'r') as f:
            val_indices = {int(line.rstrip('\n')) for line in f}
    else:
        val_indices = []

    with open(args.nbest, 'r') as f:
        for line in f:
            if ' ||| ' not in line:
                continue
            fields = line.rstrip('\n').split(' ||| ')
            sntno = int(fields[0])
            score = sacrebleu.sentence_bleu(fields[1], refs[sntno], smooth_method='floor', smooth_value=.1)
            outfile = sys.stderr if sntno in val_indices else sys.stdout
            print(sntno, fields[1], score.score, sep=' ||| ', file=outfile)


if __name__ == '__main__':
    main()
