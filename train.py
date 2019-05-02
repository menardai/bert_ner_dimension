# NER for Dimension extraction using BERT

import numpy as np
import torch

import torch.utils.data as data

from torch.optim import Adam
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification, BertAdam, BertConfig

# https://github.com/chakki-works/seqeval (will install Tensorflow 1.13)
# Note: we could extract only the f1_score function and avoid installing Tensorflow
from seqeval.metrics import f1_score

from tqdm import trange

from dataset import DimensionDataset


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# -----------------------------

epochs = 20
learning_rate = 0.002        # 3e-5
batch_size = 1
sentence_max_tokens = 32

max_grad_norm = 1.0

run_validation_loop = True
# -----------------------------
# Dataset

# Instantiate the BERT Tokenizer
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


train_dim_dataset = DimensionDataset(tokenizer, 'ner_dimension_training_set.txt', max_tokens=sentence_max_tokens)
train_dim_dataset = torch.tensor(train_dim_dataset).type(torch.LongTensor)

train_dataloader = data.DataLoader(train_dim_dataset, batch_size=batch_size, shuffle=False)


print(f"number of training samples = {len(train_dim_dataset)}")
it = iter(train_dataloader)
batch = next(it)

print(f"batch shape: {batch.shape}")
batch = batch.permute(1, 0, 2)
print(f"batch shape after permutation: {batch.shape}")
print(f"{batch_size} first samples:" )
print(batch)


valid_dim_dataset = DimensionDataset(tokenizer, 'ner_dimension_training_set.txt', max_tokens=sentence_max_tokens)
valid_dim_dataset = torch.tensor(valid_dim_dataset).type(torch.LongTensor)

valid_dataloader = data.DataLoader(valid_dim_dataset, batch_size=batch_size, shuffle=False)


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

print(torch.cuda.get_device_name(0))


# simple accuracy on a token level comparable to the accuracy in keras.
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# -----------------------------
# TRAINING LOOP

for _ in trange(epochs, desc="Epoch"):
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
    print("Train loss: {}".format(tr_loss/nb_tr_steps))

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
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        pred_tags = [DimensionDataset.labels[p_i] for p in predictions for p_i in p]
        valid_tags = [DimensionDataset.labels[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
