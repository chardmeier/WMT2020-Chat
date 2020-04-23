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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', help='Input text file.')
    parser.add_argument('outputfile', help='File to save embeddings to.')
    parser.add_argument('-model', default='xlm-roberta-base', help='Model to create embeddings with.')
    parser.add_argument('-maxsent', type=int, help='Maximum number of sentences to process.')
    parser.add_argument('-nbest', action='store_true', help='Input is in MT n-best format.')
    args = parser.parse_args()

    tokeniser = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModel.from_pretrained(args.model).eval()

    with open(args.inputfile, 'r') as f:
        if args.nbest:
            line_gen = scan_nbest(f)
        else:
            line_gen = enumerate(line.rstrip('\n') for line in f)

        maxsent = args.maxsent if args.maxsent else sys.maxsize
        input_lines = [(i, line) for i, line in itertools.takewhile(lambda tup: tup[0] < maxsent, line_gen)]

    inputs = tokeniser.batch_encode_plus((line for i, line in input_lines),
                                         return_tensors='pt', return_attention_masks=True)
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

    word_embeddings = outputs[0]
    batchsize, _, embsize = word_embeddings.shape
    sentence_indices = torch.LongTensor([i for i, line in input_lines])
    sentence_embeddings = torch.empty(batchsize, embsize)
    for i in range(batchsize):
        snt_we = word_embeddings[i, inputs['attention_mask'][i].bool(), :]
        sentence_embeddings[i, :] = torch.mean(snt_we, dim=0)

    with open(args.outputfile, 'w') as f:
        torch.save([sentence_indices, sentence_embeddings], f)


if __name__ == '__main__':
    main()
