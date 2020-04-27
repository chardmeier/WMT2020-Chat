import argparse
import itertools
import sys
import torch
import transformers


def scan_nbest(it):
    for line in it:
        fields = line.split(' ||| ')
        if len(fields) >= 2:
            yield int(fields[0]), fields[1]


def scan_and_compress_nbest(it):
    for k, g in itertools.groupby(scan_nbest(it), lambda tup: tup[0]):
        for cand in set(line for i, line in g):
            yield k, cand


def make_batches(iterable, batchsize):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batchsize))
        if len(batch) == 0:
            return
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', help='Input text file.')
    parser.add_argument('outputfile', help='File to save embeddings to.')
    parser.add_argument('-model', default='xlm-roberta-base', help='Model to create embeddings with.')
    parser.add_argument('-maxsent', type=int, help='Maximum number of sentences to process.')
    parser.add_argument('-nbest', action='store_true', help='Input is in MT n-best format.')
    parser.add_argument('-batchsize', type=int, help='Batch size to process at one time.')
    parser.add_argument('-log-sentences', action='store_true',
                        help='Log the sentences that are getting scored to stdout.')
    parser.add_argument('-device', default='cpu', help='CUDA device to use, if any.')
    args = parser.parse_args()

    device = torch.device(args.device)

    tokeniser = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModel.from_pretrained(args.model).to(device).eval()

    with open(args.inputfile, 'r') as f:
        if args.nbest:
            line_gen = scan_and_compress_nbest(f)
        else:
            line_gen = enumerate(line.rstrip('\n') for line in f)

        maxsent = args.maxsent if args.maxsent else sys.maxsize
        input_lines = [(i, line) for i, line in itertools.takewhile(lambda tup: tup[0] < maxsent, line_gen)]

    all_indices = []
    all_embeddings = []
    with torch.no_grad():
        for batch in make_batches(input_lines, args.batchsize):
            if args.log_sentences:
                for i, line in batch:
                    print('%d ||| %s' % (i, line))
            inputs = tokeniser.batch_encode_plus((line for i, line in batch), pad_to_max_length=True,
                                                 return_tensors='pt', return_attention_masks=True)
            outputs = model(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))

            word_embeddings = outputs[0]
            batchsize, _, embsize = word_embeddings.shape
            sentence_indices = torch.LongTensor([i for i, line in batch])
            sentence_embeddings = torch.empty(batchsize, embsize)
            for i in range(batchsize):
                snt_we = word_embeddings[i, inputs['attention_mask'][i].bool(), :]
                sentence_embeddings[i, :] = torch.mean(snt_we, dim=0).to('cpu')
            all_indices.append(sentence_indices)
            all_embeddings.append(sentence_embeddings)

    all_indices = torch.cat(all_indices)
    all_embeddings = torch.cat(all_embeddings)

    with open(args.outputfile, 'wb') as f:
        torch.save([all_indices, all_embeddings], f)


if __name__ == '__main__':
    main()
