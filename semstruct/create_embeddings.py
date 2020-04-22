import argparse
import itertools
import torch
import transformers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', help='Input text file.')
    parser.add_argument('outputfile', help='File to save embeddings to.')
    parser.add_argument('-model', default='xlm-roberta-base', help='Model to create embeddings with.')
    parser.add_argument('-maxsent', type=int, help='Maximum number of sentences to process.')
    args = parser.parse_args()

    tokeniser = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModel.from_pretrained(args.model).eval()

    with open(args.inputfile, 'r') as f:
        input_lines = [line.rstrip('\n') for line in itertools.islice(f, args.maxsent)]

    if args.maxsent:
        input_lines = input_lines[:args.maxsent]

    inputs = tokeniser.batch_encode_plus(input_lines, return_tensors='pt')
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

    word_embeddings = outputs[0]
    batchsize, _, embsize = word_embeddings.shape
    sentence_embeddings = torch.empty(batchsize, embsize)
    for i in range(batchsize):
        sentence_embeddings[i, :] = torch.mean(torch.masked_select(word_embeddings[i],
                                                                   inputs['attention_mask'][i].bool().unsqueeze(1)),
                                               dim=0)

    with open(args.outputfile, 'w') as f:
        torch.save(sentence_embeddings, f)


if __name__ == '__main__':
    main()
