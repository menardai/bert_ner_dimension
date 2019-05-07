import logging
from model import DimensionBertNer

logging.basicConfig(level=logging.INFO, format='%(message)s')  # no prefix

dim_ner = DimensionBertNer('models/dimension_ner_bert_best.pt')

dim_ner.predict(['I would like to resize the previous image at 1024x768',
                 'Resize to 800x600',
                 'A height of 768 and a width of 1024 would be perfect',
                ])
