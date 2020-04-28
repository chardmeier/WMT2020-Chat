import argparse
import collections
import itertools
import joblib
import sklearn.cluster
import sklearn.feature_extraction
import spacy
import sys


def dialogue_to_wordlist(nlp, dialogue):
    words = []
    for turn in dialogue:
        words.extend(t.lemma_ for t in nlp(turn) if t.tag_ in ['NN', 'NNS'])
    return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', help='Corpus file to predict or train on.')
    parser.add_argument('-train', action='store_true', help='Select training mode.')
    parser.add_argument('-predict', action='store_true', help='Select prediction mode.')
    parser.add_argument('-model', help='File name for model saving or loading.')
    parser.add_argument('-subset', type=int, help='Process first N lines of the test set only.')
    args = parser.parse_args()

    if args.train == args.predict:
        print('Please specify either -train or -predict.', file=sys.stderr)
        sys.exit(1)

    dialogues = [[]]
    with open(args.corpus, 'r') as f:
        for line in itertools.islice(f, args.subset):
            if line == '\n':
                dialogues.append([])
            else:
                dialogues[-1].append(line.rstrip('\n'))
    if not dialogues[-1]:
        dialogues.pop()

    nlp = spacy.load('en', disable=['parser', 'ner'])
    wordlists = [dialogue_to_wordlist(nlp, d) for d in dialogues]

    if args.train:
        model, preds = train(wordlists)
        joblib.dump(args.model, model)
        print_dialogues(dialogues, preds)
    else:
        model = joblib.load(args.model)
        preds = predict(model, wordlists)
        print_dialogues(dialogues, preds)


def train(wordlists):
    anchor_tags = ['pizza', 'auto', 'taxi', 'cinema', 'coffee', 'dinner']
    anchors = [['pizza'], ['auto', 'car', 'repair'], ['ride'], ['movie'], ['coffee'], ['dinner', 'restaurant']]
    nanchors = len(anchors)

    vectoriser = sklearn.feature_extraction.text.TfidfVectorizer(analyzer=lambda x: x)
    x = vectoriser.fit_transform(wordlists + anchors)
    kmeans = sklearn.cluster.KMeans(n_clusters=nanchors).fit(x)

    anchor_labels = kmeans.labels_[-nanchors:]
    if len(set(anchor_labels)) != nanchors:
        print("Anchors don't map to separate classes:", list(zip(anchors, anchor_labels)), file=sys.stderr)
        sys.exit(1)
    tagmap = ['<' + t + '>' for l, t in sorted(zip(anchor_labels, anchor_tags))]

    preds = [tagmap[x] for x in kmeans.labels_[:-nanchors]]
    model = collections.namedtuple(tagmap=tagmap, vectoriser=vectoriser, kmeans=kmeans)

    return model, preds


def predict(model, wordlists):
    x = model.vectoriser.transform(wordlists)
    labels = model.kmeans.predict(x)
    return [model.tagmap[x] for x in labels]


def print_dialogues(dialogues, preds):
    for label, dialogue in zip(preds, dialogues):
        for turn in dialogue:
            print(label, turn)
        print()


if __name__ == '__main__':
    main()
