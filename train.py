# NER for Dimension extraction using BERT
import logging
import torch

import numpy as np
import torch.utils.data as data

from torch.optim import Adam
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification

# https://github.com/chakki-works/seqeval (will install Tensorflow 1.13)
# Note: we could extract only the f1_score function and avoid installing Tensorflow
from seqeval.metrics import f1_score, classification_report
from tqdm import trange

from dataset import DimensionDataset


def get_data_loaders(tokenizer,
                     train_dataset_filename,
                     valid_dataset_filename,
                     train_batch_size = 10,
                     valid_batch_size = 50,
                     max_tokens=16):
    """
    Create instance of dataset and data loader for the given training and validation text files.

    :param tokenizer: Bert tokenizer
    :param train_dataset_filename: text file with one sample per line
    :param valid_dataset_filename: text file with one sample per line
    :param max_tokens: Bert maximum token in a sentence
    :return: datasets and data loaders for training and validation
    """
    train_dataset = DimensionDataset(tokenizer, train_dataset_filename, max_tokens=max_tokens)
    train_dataset_tensor = torch.tensor(train_dataset).type(torch.LongTensor)
    train_dataloader = data.DataLoader(train_dataset_tensor, batch_size=train_batch_size, shuffle=True)

    valid_dataset = DimensionDataset(tokenizer, valid_dataset_filename, max_tokens=max_tokens)
    valid_dataset_tensor = torch.tensor(valid_dataset).type(torch.LongTensor)
    valid_dataloader = data.DataLoader(valid_dataset_tensor, batch_size=valid_batch_size, shuffle=False)

    return train_dataset, valid_dataset, train_dataloader, valid_dataloader


def get_bert_for_token_classification(learning_rate, num_labels, is_full_finetunning=True):
    """
    Create an instance of BERT model for token classification and
    setup this model for finetuning.
    """

    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    model.cuda()

    if is_full_finetunning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        # finetune only the linear classifier on top
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

    return model, optimizer


def flat_accuracy(preds, labels):
    """ Simple accuracy on a token level comparable to the accuracy in keras. """
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_labels(predicted_classes, ground_truth_classes):
    """ Return labels for the given class numbers. """
    pred_tags = [DimensionDataset.labels[p_i] for p in predicted_classes for p_i in p]
    valid_tags = [DimensionDataset.labels[l_ii] for l in ground_truth_classes for l_i in l for l_ii in l_i]

    return pred_tags, valid_tags


def print_mislabeled_samples(dataset, pred_tags, print_correct=False):
    succeed_list = []
    failed_list = []

    for index in range(len(dataset)):
        true_dim = dataset.get_item_dimension(index)

        f = index * dataset.max_tokens
        to = f + dataset.max_tokens
        pred = pred_tags[f:to]
        pred_dim = dataset.get_item_dimension(index, pred)

        output_str = f"{true_dim}\t{pred_dim}\t{dataset.samples[index]['original']}"
        if pred_dim == true_dim:
            succeed_list.append(output_str)
        else:
            failed_list.append(output_str)

    logging.info(f"{len(failed_list)}/{len(dataset)} failed and "
                 f"{len(succeed_list)}/{len(dataset)} succeed "
                 f"({(len(succeed_list)/len(dataset)):.2f})")
    for failed in failed_list:
        logging.info(f"* {failed}")

    if print_correct:
        logging.info('----------------------')
        for succeed in succeed_list:
            logging.info(succeed)


def train(model, train_dataloader, valid_dataloader=None, nb_epochs=10,
          save_filename=None, save_min_f1_score=0.90,
          printing_f1_score_only=True, valid_dataset=None):
    """ Train the model for the specified number of epochs. """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_f1_score = 0

    for ep in trange(nb_epochs, desc="Epoch"):
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
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

            # update parameters
            optimizer.step()
            model.zero_grad()

        # print train loss per epoch
        if not printing_f1_score_only:
            logging.info("Train loss: {}".format(tr_loss/nb_tr_steps))

        if valid_dataloader:
            model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions_ids, true_label_ids = [], []

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

                predictions_ids.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_label_ids.append(label_ids)

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += b_input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss/nb_eval_steps

            if not printing_f1_score_only:
                logging.info("Validation loss: {}".format(eval_loss))
                logging.info("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

            pred_tags, valid_tags = get_labels(predictions_ids, true_label_ids)

            score = f1_score(pred_tags, valid_tags)

            if save_filename and score > best_f1_score and score > save_min_f1_score:
                best_f1_score = score
                torch.save(model.state_dict(), save_filename)
                logging.info(f"\tF1-Score: {score :.5f} \t(model saved)")
            else:
                logging.info(f"\tF1-Score: {score :.5f}")

            if ep == nb_epochs-1:
                logging.info("")
                logging.info(classification_report(pred_tags, valid_tags))

                if valid_dataset:
                    print_mislabeled_samples(valid_dataset, pred_tags)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')  # no prefix

    # Instantiate the BERT Tokenizer. Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # train_batch_size is an important hyper params in fine tuning this model
    _, valid_dataset, train_dataloader, valid_dataloader = get_data_loaders(tokenizer,
                                                          "data/ner_dimension_training_set2.txt",
                                                          "data/ner_dimension_valid_set.txt",
                                                          train_batch_size=10)

    model, optimizer = get_bert_for_token_classification(learning_rate = 2e-5,
                                                         num_labels=len(DimensionDataset.label2idx))

    train(model, train_dataloader, valid_dataloader, nb_epochs=30, valid_dataset=valid_dataset,
          save_filename='models/dimension_ner_bert.pt',
          save_min_f1_score=0.95)
