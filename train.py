# NER for Dimension extraction using BERT

import numpy as np
import torch

import torch.utils.data as data

from torch.optim import Adam
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification, BertAdam, BertConfig

# https://github.com/chakki-works/seqeval (will install Tensorflow 1.13)
# Note: we could extract only the f1_score function and avoid installing Tensorflow
from seqeval.metrics import f1_score, classification_report

from tqdm import trange

from dataset import DimensionDataset


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)  # include module/function call at prefix
logging.basicConfig(level=logging.INFO, format='%(message)s')  # no prefix

# -----------------------------

epochs = 20
learning_rate = 0.0005        # 3e-5
train_batch_size = 1
valid_batch_size = 100
sentence_max_tokens = 16

max_grad_norm = 1.0

train_dataset_filename = "ner_dimension_training_set2.txt"
#train_dataset_filename = "ner_dimension_training_set.txt"
#train_dataset_filename = "ner_dimension_training_set_easy_2.txt"

valid_dataset_filename = "ner_dimension_valid_set.txt"

run_validation_loop = True
printing_f1_score_only = True

# -----------------------------
# Dataset

# Instantiate the BERT Tokenizer
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dim_dataset = DimensionDataset(tokenizer, train_dataset_filename, max_tokens=sentence_max_tokens)
train_dim_dataset = torch.tensor(train_dim_dataset).type(torch.LongTensor)
train_dataloader = data.DataLoader(train_dim_dataset, batch_size=train_batch_size, shuffle=True)

valid_dim_dataset = DimensionDataset(tokenizer, valid_dataset_filename, max_tokens=sentence_max_tokens)
valid_dim_dataset = torch.tensor(valid_dim_dataset).type(torch.LongTensor)
valid_dataloader = data.DataLoader(valid_dim_dataset, batch_size=valid_batch_size, shuffle=False)


logging.info(f"number of training samples = {len(train_dim_dataset)}")
it = iter(train_dataloader)
batch = next(it)

logging.info(f"batch shape: {batch.shape}")
batch = batch.permute(1, 0, 2)
logging.info(f"batch shape after permutation: {batch.shape}")
logging.info(f"{train_batch_size} first samples:" )
logging.info(batch)


# -----------------------------
# Setup BERT Model for Finetuning
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(DimensionDataset.label2idx))
model.cuda()

# finetune only the linear classifier on top
param_optimizer = list(model.classifier.named_parameters())
optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)


# -----------------------------
# Finetune BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

logging.info(torch.cuda.get_device_name(0))


# simple accuracy on a token level comparable to the accuracy in keras.
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# -----------------------------
# TRAINING LOOP

def get_labels(predictions, true_labels, filter_f1_labels):
    pred_tags = [DimensionDataset.labels[p_i] for p in predictions for p_i in p]
    valid_tags = [DimensionDataset.labels[l_ii] for l in true_labels for l_i in l for l_ii in l_i]

    if filter_f1_labels:
        # filter both predictions array to keep only the ones from true_labels with non zeros
        np_true_labels = np.array(valid_tags)
        np_predictions = np.array(pred_tags)

        mask = np.ma.masked_where(np_true_labels == 'O', np_true_labels)
        valid_tags = np.ma.compressed(mask).tolist()

        mask = np.ma.masked_where(np_true_labels == 'O', np_predictions)
        pred_tags = np.ma.compressed(mask).tolist()

    return pred_tags, valid_tags

for ep in trange(epochs, desc="Epoch"):
    # *** TRAIN LOOP ***
    model.train()

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):
        # permute the tensor to go from shape (batch size, 3, max_tokens) to (3, batch size, max tokens)
        batch = batch.permute(1, 0, 2)

        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask,
                     labels=b_labels)

        # backward pass
        loss.backward()

        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

        # update parameters
        optimizer.step()
        model.zero_grad()

    # print train loss per epoch
    if not printing_f1_score_only:
        logging.info("Train loss: {}".format(tr_loss/nb_tr_steps))

    if run_validation_loop:
        # *** VALIDATION LOOP ***
        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []

        for batch in valid_dataloader:
            batch = batch.permute(1, 0, 2)
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)

                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss/nb_eval_steps

        if not printing_f1_score_only:
            logging.info("Validation loss: {}".format(eval_loss))
            logging.info("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        pred_tags, valid_tags = get_labels(predictions, true_labels, filter_f1_labels=False)
        f_pred_tags, f_valid_tags = get_labels(predictions, true_labels, filter_f1_labels=True)

        logging.info(f"\tF1-Score: {f1_score(pred_tags, valid_tags):.5f}"
                     f"\tF1-Score (filtered): {f1_score(f_pred_tags, f_valid_tags):.5f}")

        if ep == epochs-1 or (ep != 0 and ep % 10 == 0):
            logging.info("")
            logging.info(classification_report(pred_tags, valid_tags))
