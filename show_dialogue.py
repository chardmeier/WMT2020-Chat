import json
import random


def main():
    with open('BConTrasT/train.json', 'r') as f:
        train = json.load(f)

    # dialogue_id = 'dlg-04990162-b3b4-40c1-8ffb-0497648e83ff'
    # dialogue_id = 'dlg-e1e4c900-61e2-4ba2-8772-064ed58273a3'
    dialogue_id = random.sample(train.keys(), 1)[0]

    print('Dialogue ID: ' + dialogue_id + '\n')
    for utt in train[dialogue_id]:
        utt['english'] = utt['source' if utt['speaker'] == 'agent' else 'target']
        utt['german'] = utt['target' if utt['speaker'] == 'agent' else 'source']
        print('{speaker}: {german}'.format(**utt))

    print('\n')
    for utt in train[dialogue_id]:
        print('{speaker}: {english}'.format(**utt))


if __name__ == '__main__':
    main()