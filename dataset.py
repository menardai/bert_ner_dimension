import re

import numpy as np
import torch.utils.data as data


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
            sample['attention_mask'] = [float(i>0) for i in sample['input_ids']]
            
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

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample['input_ids'], sample['attention_mask'], sample['labels_ids']
    
    def __len__(self):
        return len(self.samples)

