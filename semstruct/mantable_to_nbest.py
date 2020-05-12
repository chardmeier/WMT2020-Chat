import argparse
import pandas
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('table', help='Manual evaluation spreadsheet (TSV).')
    args = parser.parse_args()

    with open(args.table, 'r') as f:
        table = pandas.read_table(f, header=0)

    ref_col = 2
    first_sys = 6

    for i, snt in table.iterrows():
        if snt['Speaker'] != 'agent':
            continue

        pos = [ref_col] + list(range(first_sys, len(snt), 2))
        cands = set((snt.iloc[j], snt.iloc[j + 1]) for j in pos if not pandas.isna(snt.iloc[j + 1]))

        seen = set()
        discard = set()
        for t, c in cands:
            if t in seen:
                discard.add(t)
            seen.add(t)
        if discard:
            print('Discarding conflicting annotations in sentence', i, file=sys.stderr)
            for c in discard:
                print(c, file=sys.stderr)
            cands = [(t, c) for t, c in cands if c not in discard]

        if len(cands) > 1:
            for t, c in cands:
                print('%d ||| %s ||| %d' % (i, t, c))


if __name__ == '__main__':
    main()
