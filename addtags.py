import argparse
import re


def addressee_reference(srctok, tgttok):
    pat_you = re.compile(r'[Yy]ou')
    pat_they = re.compile(r'[Tt]he[ym]')
    pat_du = re.compile(r'[Dd](u|ich|ir)')
    pat_sie = re.compile(r'Sie|Ihnen')

    s_you = sum(1 for t in srctok if pat_you.fullmatch(t))
    s_they = sum(1 for t in srctok if pat_they.fullmatch(t))

    t_du = sum(1 for t in tgttok if pat_du.fullmatch(t))
    t_sie = sum(1 for t in tgttok if pat_sie.fullmatch(t))

    if s_you > 0 and t_du > 0 and t_sie == 0:
        return ['<POL:du>']
    elif s_you > 0 and s_they == 0 and t_sie > 0:
        return ['<POL:Sie>']
    else:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='Tokenised source corpus')
    parser.add_argument('tgt', help='Tokenised target corpus')
    args = parser.parse_args()

    with open(args.src, 'r') as src, open(args.tgt, 'r') as tgt:
        for srcline, tgtline in zip(src, tgt):
            srctok = srcline.rstrip('\n').split(' ')
            tgttok = tgtline.rstrip('\n').split(' ')
            tags = addressee_reference(srctok, tgttok)
            if tags:
                print(tags)
                print(srcline, end='')
                print(tgtline, end='')
                print()


if __name__ == '__main__':
    main()
