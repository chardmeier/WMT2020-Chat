import collections
import json


def main():
    bcontrast = 'BConTrasT/dev.json'
    fnames = {
        'customer': 'translations-fair/dev-t-ensemble.de-en.en',
        'agent': 'translations-fair/dev-t-ensemble.en-de.de'
    }

    with open(bcontrast, 'r') as f:
        dev = json.load(f, object_pairs_hook=collections.OrderedDict)

    translations = {}
    for speaker, fname in fnames.items():
        with open(fname, 'r') as f:
            translations[speaker] = [line.rstrip('\n') for line in f]

    for dialogue_id, dialogue in dev.items():
        for utt in dialogue:
            utt['translation'] = translations[utt['speaker']].pop(0)

        print_dialogue(dialogue_id, dialogue)


def print_dialogue(dialogue_id, dialogue):
    print('Dialogue', dialogue_id)
    for utt in dialogue:
        print('{speaker}: {source}\n    [RF] {target}\n    [MT] {translation}'.format(**utt))
    print()


if __name__ == '__main__':
    main()
