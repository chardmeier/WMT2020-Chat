import json


def load_training(self_dialogs_file, dataset_file):
    #self_dialogs_file = 'Taskmaster/TM-1-2019/self-dialogs.json'
    #train_csv = 'Taskmaster/TM-1-2019/train-dev-test/train.csv'

    with open(self_dialogs_file, 'r') as f:
        self_dialogs = json.load(f)

    with open(dataset_file, 'r') as f:
        train_ids = {line.rstrip('\n').split(',')[0] for line in f}

    train_self = [d for d in self_dialogs if d['conversation_id'] in train_ids]

    return train_self


def print_dialogue(d):
    for u in d['utterances']:
        print('{speaker}\t{text}'.format(**u))
    print()


def main(self_dialogs_file, dataset_file):
    train = load_training(self_dialogs, dataset)

    for d in train[:10]:
        print_dialogue(d)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('self_dialogs_file')
    parser.add_argument('dataset_file')
    args = parser.parse_args()

    main(args.self_dialogs_file, args.dataset_file)
