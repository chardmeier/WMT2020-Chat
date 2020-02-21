import json


def load_training():
    self_dialogs_file = 'Taskmaster/TM-1-2019/self-dialogs.json'
    train_csv = 'Taskmaster/TM-1-2019/train-dev-test/train.csv'

    with open(self_dialogs_file, 'r') as f:
        self_dialogs = json.load(f)

    with open(train_csv, 'r') as f:
        train_ids = {line.rstrip('\n').split(',')[0] for line in f}

    train_self = [d for d in self_dialogs if d['conversation_id'] in train_ids]

    return train_self


def print_dialogue(d):
    for u in d['utterances']:
        print('{speaker}\t{text}'.format(**u))
    print()


def main():
    train = load_training()

    for d in train[:10]:
        print_dialogue(d)


if __name__ == '__main__':
    main()
