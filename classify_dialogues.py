import argparse
import collections
import contextlib
import itertools
import joblib
import json
import scipy.sparse
import sklearn.cluster
import sklearn.decomposition
import sklearn.feature_extraction
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
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
    parser.add_argument('-truncate', type=int, help='Number of dialogue-initial sentences to train on.')
    parser.add_argument('-lsa', type=int, help='Use LSA for dimensionality reduction to N dimensions.')
    parser.add_argument('-json', action='store_true', help='Input and output BConTrasT JSON files.')
    parser.add_argument('-eval', action='store_true', help='Evaluate on examples having a gold_dialogue_type.')
    args = parser.parse_args()

    if args.train == args.predict:
        print('Please specify either -train or -predict.', file=sys.stderr)
        sys.exit(1)

    if args.json:
        with open(args.corpus, 'r') as f:
            in_json = json.load(f, object_pairs_hook=collections.OrderedDict)
        srctgt = {'customer': 'target', 'agent': 'source'}
        if args.eval:
            def utterances(d):
                return d['utterances']
        else:
            def utterances(d):
                return d
        dialogues = [[t[srctgt[t['speaker']]] for t in utterances(d)[:args.truncate]] for d in in_json.values()]
    else:
        dialogues = [[]]
        with open(args.corpus, 'r') as f:
            for line in itertools.islice(f, args.subset):
                if line == '\n':
                    dialogues.append([])
                else:
                    if len(dialogues[-1]) < args.truncate:
                        dialogues[-1].append(line.rstrip('\n'))
        if not dialogues[-1]:
            dialogues.pop()

    nlp = spacy.load('en', disable=['parser', 'ner'])
    wordlists = [dialogue_to_wordlist(nlp, d) for d in dialogues]

    if args.train:
        if args.eval:
            train_wordlists = [w
                               for t, w in zip((dict(d).get('gold_dialogue_type') for d in in_json.values()), wordlists)
                               if t is not None]
        else:
            train_wordlists = wordlists
        model, preds = train(train_wordlists, lsa=args.lsa)
        joblib.dump(model, args.model)
    else:
        model = joblib.load(args.model)
        preds = predict(model, wordlists)

    if args.json:
        out = [(k, {'dialogue_type': p, 'utterances': u}) for p, (k, u) in zip(preds, in_json.items())]
        print(json.dumps(collections.OrderedDict(out), indent=4))
    else:
        print_dialogues(dialogues, preds)

    if args.eval:
        gold_tags = [d.get('gold_dialogue_type') for d in in_json.values()]
        labels = list(sorted(set(gold_tags).difference({None}).union(preds)))
        gold_pairs = [(p, g) for p, g in zip(preds, gold_tags) if g is not None]
        gold = [g for p, g in gold_pairs]
        pred = [p for p, g in gold_pairs]
        p, r, f, s = sklearn.metrics.precision_recall_fscore_support(gold, pred, labels=labels)
        with contextlib.redirect_stdout(sys.stderr):
            print('                   P      R      F    Support')
            for i, l in enumerate(labels):
                print('%-15s  %0.3f  %0.3f  %0.3f  %5d' % (l, p[i], r[i], f[i], s[i]))
            print('%-15s  %0.3f  %0.3f  %0.3f' % ('MACRO', p.mean(), r.mean(), f.mean()))
            print()
            cmat = sklearn.metrics.confusion_matrix(gold, pred, labels=labels)
            print_confusion_matrix(labels, cmat)


def print_confusion_matrix(labels, matrix):
    print('              predicted as:')
    print('gold label: ', end='')
    for lb in labels:
        print('  %10s' % lb, end='')
    print()
    for i, lb in enumerate(labels):
        print('%-12s' % lb, end='')
        for j, lb2 in enumerate(labels):
            print('  %9d ' % matrix[i, j], end='')
        print()


def identity(x):
    return x


def make_dense(mat):
    if scipy.sparse.isspmatrix(mat):
        return mat.toarray()
    else:
        return mat


def train(wordlists, lsa=None):
    anchor_tags = ['pizza', 'auto', 'taxi', 'cinema', 'coffee', 'dinner']
    anchors = [['pizza'], ['auto', 'car', 'repair'], ['ride'], ['movie'], ['coffee'], ['dinner', 'restaurant']]
    nanchors = len(anchors)

    steps = [sklearn.feature_extraction.text.TfidfVectorizer(analyzer=identity)]

    if lsa:
        steps.append(sklearn.decomposition.TruncatedSVD(lsa))
        steps.append(sklearn.preprocessing.Normalizer(copy=False))

    pipeline = sklearn.pipeline.make_pipeline(*steps)
    x = pipeline.fit_transform(wordlists + anchors)
    kmeans = sklearn.cluster.KMeans(n_clusters=nanchors, init=make_dense(x[-nanchors:]), n_init=1).fit(x)

    anchor_labels = kmeans.labels_[-nanchors:]
    if len(set(anchor_labels)) != nanchors:
        print("Anchors don't map to separate classes:", list(zip(anchors, anchor_labels)), file=sys.stderr)
        sys.exit(1)
    tagmap = ['<' + t + '>' for l, t in sorted(zip(anchor_labels, anchor_tags))]

    preds = [tagmap[x] for x in kmeans.labels_[:-nanchors]]
    model = {'tagmap': tagmap, 'pipeline': pipeline, 'kmeans': kmeans}

    return model, preds


def predict(model, wordlists):
    x = model['pipeline'].transform(wordlists)
    labels = model['kmeans'].predict(x)
    return [model['tagmap'][x] for x in labels]


def print_dialogues(dialogues, preds):
    for label, dialogue in zip(preds, dialogues):
        for turn in dialogue:
            print(label, turn)
        print()


if __name__ == '__main__':
    main()
