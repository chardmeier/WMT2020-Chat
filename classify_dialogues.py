import itertools
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
    dialogues = [[]]
    with open('train-fulldialogues.en', 'r') as f:
        for line in f:
            if line == '\n':
                dialogues.append([])
            else:
                dialogues[-1].append(line.rstrip('\n'))
    if not dialogues[-1]:
        dialogues.pop()

    nlp = spacy.load('en', disable=['parser', 'ner'])
    wordlists = [dialogue_to_wordlist(nlp, d) for d in dialogues]

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
    for l, d in zip(kmeans.labels_, dialogues):
        label = tagmap[l]
        for turn in d:
            print(label, turn)
        print()


if __name__ == '__main__':
    main()
