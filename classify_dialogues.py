import itertools
import sklearn.cluster
import sklearn.feature_extraction
import spacy


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

    vectoriser = sklearn.feature_extraction.text.TfidfVectorizer(analyzer=lambda x: x)
    x = vectoriser.fit_transform(wordlists)
    kmeans = sklearn.cluster.KMeans(n_clusters=6).fit(x)
    for i, d in enumerate(dialogues):
        label = '<%d>' % kmeans.labels_[i]
        for turn in d:
            print(label, turn)
        print()


if __name__ == '__main__':
    main()
