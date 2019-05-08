import logging
from model import DimensionBertNer

logging.basicConfig(level=logging.INFO, format='%(message)s')  # no prefix

dim_ner = DimensionBertNer('models/dimension_ner_bert_best.pt')

dialog_sentences = ['I would like to resize the previous image at 1024x768',
                    'Resize to 800 and 600',
                    'A height of 768 and a width of 1024',
                    ]

dims = dim_ner.predict(dialog_sentences)

for sentence, dim in zip(dialog_sentences, dims):
    print(f"{sentence} \t {dim}")
