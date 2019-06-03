# NER for Dimension extraction using BERT
import logging
import re
import torch

import numpy as np
import torch.utils.data as data

from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification, BertConfig


class DimensionDataset(data.Dataset):

    labels = ['O', 'W', 'H', 'U']
    label2idx = {label: i for i, label in enumerate(labels)}

    def __init__(self, tokenizer, dataset_filename=None, lines_to_predict=None, max_tokens=24):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        # make sure the index of label "O" (other) is 0 since we will pad with 0 later
        assert DimensionDataset.label2idx.get('O') == 0

        # make sure at least one of dataset_filename or lines_to_predict is not none
        if dataset_filename is None and lines_to_predict is None:
            raise ValueError('Both dataset_filename and lines_to_predict are undefined, please provide one of them.')

        # make sure only one of dataset_filename and lines_to_predict is not none
        if dataset_filename and lines_to_predict:
            raise ValueError('Please provide only one of these two params: dataset_filename and lines_to_predict.')

        self.samples = []

        if dataset_filename:
            with open(dataset_filename) as f:
                lines = [line.rstrip('\n') for line in f if line.strip()]

        if lines_to_predict:
            lines = [line.rstrip('\n') for line in lines_to_predict if line.strip()]

        for line in lines:
            # ex: "Resize 1 image to 640x480"
            sample = {'original': line}

            # store a list of all numbers in this text line
            # ex: ['1', '640', '480']
            sample['number_list'] = re.findall('[0-9]+', line, re.IGNORECASE)

            # replace all numbers by the string " number "
            # ex: "Resize  number  image to  number x number "
            text_number = re.sub('[0-9]+', ' number ', line)

            # replace equivalent tokens to reduce the number of training examples
            # ex: returns "Resize  number  image to  number x number "
            #     from "Resize  number  image to ( number , number )"
            text_number = DimensionDataset.optimize_tokens(text_number)

            # store the tokenized version
            # ex: ['[CLS]', 'res', '##ize', 'number', 'image', 'to', 'number', 'x', 'number']
            sample['tokenized_text'] = tokenizer.tokenize("[CLS] " + text_number)

            # convert tokens into indexes (embeddings)
            sample['input_ids'] = self.zero_padding(
                tokenizer.convert_tokens_to_ids(sample['tokenized_text']))

            # create a mask to ignore the padded elements in the sequences
            sample['attention_mask'] = [float(i > 0) for i in sample['input_ids']]

            # extract truth label from original text for given dataset.
            # (in case of text_to_predict, the labels will be compute using DimensionBertNer.predict() function)
            if dataset_filename:
                # store the tag for each token
                # ex: ['O', 'O', 'O', 'O', 'O', 'O', 'W', 'O', 'H']
                sample['labels'] = self._get_labels(sample['tokenized_text'], sample['number_list'])

                assert len(sample['tokenized_text']) == len(sample['labels'])

                # convert labels in indexes
                sample['labels_ids'] = self.zero_padding(
                    [DimensionDataset.label2idx.get(label) for label in sample['labels']])

                assert len(sample['input_ids']) == len(sample['labels_ids'])
            else:
                # store all zeros in labels since we don't know them
                sample['labels_ids'] = np.zeros(self.max_tokens)

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

    def _get_labels(self, tokenized_text, num_list):
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

    def set_labels(self, predictions_ids, predictions_labels):
        for index, (idx, label) in enumerate(zip(predictions_ids, predictions_labels)):
            self.samples[index]['labels_ids'] = idx
            self.samples[index]['labels'] = label

    @staticmethod
    def optimize_tokens(text_number):
        """
        replace equivalent tokens to reduce the number of training examples.


        ex: from "Resize  number  image to ( number , number )"
            returns "Resize  number  image to  number x number "
        """
        # "(number,number)" --> "number x number"
        text_number = re.sub('\((\s*)number(\s*),(\s*)number(\s*)\)', ' number x number ', text_number)

        # "(number x number)"  --> "number x number"
        text_number = re.sub('(\(+)(\s*)number(\s*)x(\s*)number(\s*)(\)+)', ' number x number ', text_number)

        #  "number/number"  --> "number x number"
        # ("number/number") --> "number x number"
        text_number = re.sub('(\(*)(\s*)number(\s*)/(\s*)number(\s*)(\)*)', ' number x number ', text_number)

        #  "number:number"  --> "number x number"
        # ("number:number") --> "number x number"
        text_number = re.sub('(\(*)(\s*)number(\s*):(\s*)number(\s*)(\)*)', ' number x number ', text_number)

        return text_number

    @staticmethod
    def dimension(tokenized_text, labels, number_list):
        '''
        Return a dict with the dimension info extracted from given params.
        ex: {'W': 640, 'H':480}
        '''
        dim = {}
        if number_list:
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
        # note: all three list returned must have the dimension
        return sample['input_ids'], sample['attention_mask'], sample['labels_ids']

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        str = f"{self.__class__} - {len(self.samples)} samples\n"

        for index, sample in enumerate(self.samples):
            str += f"index {index}\n"
            str += f"\toriginal:\t{sample['original']}\n"
            str += f"\tnumber_list:\t{sample['number_list']}\n"
            str += f"\ttokenized_text:\t{sample['tokenized_text']}\n"
            str += f"\tinput_ids:\t{sample['input_ids']}\n"
            str += f"\tattention_mask:\t{sample['attention_mask']}\n"
            str += f"\tlabels_ids:\t{sample['labels_ids']}\n"
            if sample['labels']:
                str += f"\tlabels:\t{sample['labels']}\n"
            else:
                str += f"\tlabels:\tundefined\n"

        return str


class DimensionBertNer(object):

    def __init__(self, model_weight_filename=None):
        """
        Load an instance of BERT model for dimension classification.
        """
        self.num_labels = len(DimensionDataset.label2idx)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info('*** Instantiate model ***')
        if model_weight_filename:
            config = BertConfig(vocab_size_or_config_json_file=30522, hidden_size=768,
                                num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

            self.model = BertForTokenClassification(config, self.num_labels)

            logging.info('*** Loading model weights ***')
            self.model.load_state_dict(torch.load(model_weight_filename, map_location=self.device))
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

    def predict(self, lines_to_predict, max_tokens=24):
        """
        Returns dimension dict for each text line of the given lines_to_predict param.

        :param lines_to_predict: list of text to decode
        :param max_tokens: maximum tokens per sentence
        :return: Return dimension for the specified sample index. ex: {'W': 640, 'H':480}
        """
        # build the data loader
        bs = min(64, len(lines_to_predict))
        dataset = DimensionDataset(self.tokenizer, lines_to_predict=lines_to_predict, max_tokens=max_tokens)
        dataset_tensor = torch.tensor(dataset).type(torch.LongTensor)
        dataloader = data.DataLoader(dataset_tensor, batch_size=bs, shuffle=False)

        self.model.to(self.device)
        self.model.eval()

        predictions_ids = []

        for batch in dataloader:
            # permute the tensor to go from shape (batch size, 3, max_tokens) to (3, batch size, max tokens)
            batch = batch.permute(1, 0, 2)

            # add batch to gpu
            batch = tuple(t.to(self.device) for t in batch)
            batch_input_ids, batch_input_mask, _ = batch

            with torch.no_grad():
                logits = self.model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)

            logits = logits.detach().cpu().numpy()

            predictions_ids.extend([list(p) for p in np.argmax(logits, axis=2)])

        # convert prediction indexes in labels. Resulting in a list of shape [nb_samples, max_tokens]
        predictions_labels = [[DimensionDataset.labels[class_idx] for class_idx in pred] for pred in predictions_ids]

        # set the predicted labels (class id and label)
        dataset.set_labels(predictions_ids, predictions_labels)
        #logging.info(dataset)

        predicted_dim = [dataset.get_item_dimension(i) for i in range(len(dataset))]
        #logging.info(predicted_dim)

        return predicted_dim
