import argparse
import sacrebleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', help='Reference translation.')
    parser.add_argument('nbest', help='N-best list to score.')
    args = parser.parse_args()

    with open(args.reference, 'r') as f:
        refs = [line.rstrip('\n') for line in f]

    with open(args.nbest, 'r') as f:
        for line in f:
            if ' ||| ' not in line:
                continue
            fields = line.rstrip('\n').split(' ||| ')
            sntno = int(fields[0])
            score = sacrebleu.sentence_bleu(fields[1], refs[sntno], smooth_method='floor', smooth_value=.1)
            print(sntno, fields[1], score.score, sep=' ||| ')


if __name__ == '__main__':
    main()
