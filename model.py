# NER for Dimension extraction using BERT
import logging
import re
import torch

import numpy as np
import torch.utils.data as data

from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification


class DimensionDataset(data.Dataset):
    labels = ['O', 'W', 'H', 'U']
    label2idx = {label: i for i, label in enumerate(labels)}

    def __init__(self, tokenizer, dataset_filename, max_tokens=16):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        # make sure the index of label "O" (other) is 0 since we will pad with 0 later
        assert DimensionDataset.label2idx.get('O') == 0

        self.samples = []

        with open(dataset_filename) as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]

        for line in lines:
            sample = {}

            # ex: "Resize 1 image to 640x480"
            sample['original'] = line

            # store a list of all numbers in this text line
            # ex: ['1', '640', '480']
            sample['number_list'] = re.findall('[0-9]+', line, re.IGNORECASE)

            # replace all numbers by the string " number "
            # ex: "Resize  number  image to  number x number "
            text_number = re.sub('[0-9]+', ' number ', line)

            # store the tokenized version
            # ex: ['[CLS]', 'res', '##ize', 'number', 'image', 'to', 'number', 'x', 'number']
            sample['tokenized_text'] = tokenizer.tokenize("[CLS] " + text_number)

            # store the tag for each token
            # ex: ['O', 'O', 'O', 'O', 'O', 'O', 'W', 'O', 'H']
            sample['labels'] = self._labels(sample['tokenized_text'], sample['number_list'])

            assert len(sample['tokenized_text']) == len(sample['labels'])

            # convert tokens into indexes (embeddings)
            sample['input_ids'] = self.zero_padding(
                tokenizer.convert_tokens_to_ids(sample['tokenized_text']))

            # convert labels in indexes
            sample['labels_ids'] = self.zero_padding(
                [DimensionDataset.label2idx.get(label) for label in sample['labels']])

            assert len(sample['input_ids']) == len(sample['labels_ids'])

            # create a mask to ignore the padded elements in the sequences
            sample['attention_mask'] = [float(i > 0) for i in sample['input_ids']]

            self.samples.append(sample)

    def zero_padding(self, a):
        ''' Add zeros at the end of array "a" to reach max_tokens length. Trunk if "a" is too large. '''
        if len(a) > self.max_tokens:
            return a[:self.max_tokens]
        else:
            padded = np.zeros(self.max_tokens)
            n = np.array(a)

            padded[:len(a)] = n
            return padded.astype(np.int32).tolist()

    def _labels(self, tokenized_text, num_list):
        '''
        Generate labels for each token: 640->W, 480->H, 800->U, *->O

        Returns:
          for this input string: "Resize 1 image to 640x480"
          tokenized version:     ['[CLS]', 'res', '##ize', 'number', 'image', 'to', 'number', 'x', 'number']
          the result would be:   [  'O',    'O',    'O',     'O',      'O',   'O',    'W',    'O',   'H']
        '''
        labels = []
        num_index = 0
        for token in tokenized_text:
            if token == 'number':
                if num_list[num_index] == '640':
                    labels.append('W')
                elif num_list[num_index] == '480':
                    labels.append('H')
                elif num_list[num_index] == '800':
                    labels.append('U')
                else:
                    labels.append('O')

                num_index += 1
            else:
                labels.append('O')

        return labels

    @staticmethod
    def dimension(tokenized_text, labels, number_list):
        '''
        Return a dict with the dimension info extracted from given params.
        ex: {'W': 640, 'H':480}
        '''
        dim = {}
        index = 0
        for token, label in zip(tokenized_text, labels):
            if token == 'number':
                value = int(number_list[index])
                index += 1

                if label == 'W':
                    dim['W'] = value
                elif label == 'H':
                    dim['H'] = value
                elif label == 'U':
                    dim['U'] = value
        return dim

    def get_item_dimension(self, index, labels=None):
        '''
        Return dimension for the specified sample index.
        ex: {'W': 640, 'H':480}
        '''
        if not labels:
            labels = self.samples[index]['labels']

        return DimensionDataset.dimension(self.samples[index]['tokenized_text'],
                                          labels,
                                          self.samples[index]['number_list'])

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample['input_ids'], sample['attention_mask'], sample['labels_ids']

    def __len__(self):
        return len(self.samples)


class DimensionBertNer(object):

    def __init__(self, model_weight_filename=None):
        """
        Load an instance of BERT model for dimension classification .
        """
        self.num_labels = len(DimensionDataset.label2idx)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info('*** Instantiate model ***')
        if model_weight_filename:
            # todo - do not load pretrained, we only want the BertForTokenClassification model setup
            #self.model = BertForTokenClassification(config=..., num_labels=num_labels)
            self.model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=self.num_labels)

            logging.info('*** Loading model weights ***')
            self.model.load_state_dict(torch.load(model_weight_filename))
        else:
            # load bert pretrained with empty token classification top layers
            self.model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=self.num_labels)

        logging.info('*** Loading tokenizer ***')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def flat_accuracy(preds, labels):
        """ Simple accuracy on a token level comparable to the accuracy in keras. """
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)



