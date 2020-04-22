import collections


def main():
    # first_input = '/Users/christianhardmeier/Documents/project/2020-WMT-Chat/devset-checked.txt'
    # fnames = [
    #     ('ED-PC', {
    #         'customer': 'translations-uedin/ft-paracrawl+train.chat.dev.de-en.en',
    #         'agent': 'translations-uedin/ft-paracrawl+train.chat.dev.en-de.de'
    #     }),
    #     ('FB-TR', {
    #         'customer': 'translations-fair/ft-train.chat.dev.de-en.en',
    #         'agent': 'translations-fair/ft-train.chat.dev.en-de.de'
    #     })
    # ]
    first_input = '/Users/christianhardmeier/Documents/project/2020-WMT-Chat/devset-checked-subset-4trans.txt'
    fnames = [
        ('ED-TR', {
            'customer': 'translations-uedin/ft-train.chat.dev.de-en.en',
            'agent': 'translations-uedin/ft-train.chat.dev.en-de.de'
        })
    ]

    systems = collections.OrderedDict()
    for system, fn in fnames:
        translations = {}
        systems[system] = translations
        for speaker, fname in fn.items():
            with open(fname, 'r') as f:
                translations[speaker] = [line.rstrip('\n') for line in f]

    with open(first_input, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            print(line)
            if line.startswith('customer:') or line.startswith('agent:'):
                speaker = line[:line.index(':')]
                for system, translations in systems.items():
                    print('    [%s] ' % system + translations[speaker].pop(0))


if __name__ == '__main__':
    main()
