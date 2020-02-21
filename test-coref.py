import allennlp.predictors
import collections
import taskmaster


def format_dialogue_with_coref(pred):
    mention_start = collections.defaultdict(list)
    mention_end = collections.defaultdict(list)
    for i, spans in enumerate(pred['clusters']):
        for s, e in spans:
            mention_start[s].append((e, i))
            mention_end[e].append((s, i))

    for k in mention_start.keys():
        mention_start[k] = [i for e, i in sorted(mention_start[k], reverse=True)]
    for k in mention_end.keys():
        mention_end[k] = [i for s, i in sorted(mention_end[k])]

    colours = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']

    outstr = ''
    for i, tok in enumerate(pred['document']):
        for cluster_id in mention_start[i]:
            outstr += '<span style="color:%s">[' % colours[cluster_id % len(colours)]
        outstr += tok
        for cluster_id in mention_end[i]:
            outstr += ']_%d</span>' % cluster_id
        if tok in {'.', '!', '?'}:
            outstr += '<br/>\n'
        else:
            outstr += ' '

    if outstr[-6:] == '<br/>\n':
        outstr = outstr[:-6]

    return outstr


def main():
    coref_model = 'https://allennlp.s3.amazonaws.com/models/coref-model-2020.02.10.tar.gz'
    predictor = allennlp.predictors.Predictor.from_path(coref_model)

    train = taskmaster.load_training()

    outdoc = '<html><head><title>Taskmaster Examples</title></head><body>'

    for doc in train[::200]:
        doctxt = '\n'.join(u['text'] for u in doc['utterances'])
        pred = predictor.predict(document=doctxt)
        outdoc += format_dialogue_with_coref(pred)
        outdoc += '<hr/>'

    outdoc += '</body></html>'

    with open('output.html', 'w') as f:
        print(outdoc, file=f)


if __name__ == '__main__':
    main()