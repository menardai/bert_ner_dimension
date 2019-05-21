# NER for Dimension extraction using BERT
import os
import logging
import torch

import numpy as np
import torch.utils.data as data

from torch.optim import Adam

# https://github.com/chakki-works/seqeval (will install Tensorflow 1.13)
# Note: we could extract only the f1_score function and avoid installing Tensorflow
from seqeval.metrics import f1_score, classification_report
from tqdm import trange

from model import DimensionDataset, DimensionBertNer


def get_data_loaders(tokenizer,
                     train_dataset_filename,
                     valid_dataset_filename,
                     train_batch_size = 10,
                     valid_batch_size = 50,
                     max_tokens=24):
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


def setup_model_for_finetuning(model, learning_rate, is_full_finetunning=True):
    """
    Setup the specified model for finetuning, create the optimizer.
    """
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


def train(model, optimizer, train_dataloader, valid_dataloader=None, nb_epochs=10,
          save_filename=None, save_min_f1_score=0.90,
          eval_f1_score_only=True, valid_dataset=None):
    """ Train the model for the specified number of epochs. """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_f1_score = 0

    for ep in trange(nb_epochs, desc="Epoch"):
        model.train()

        train_loss = 0
        nb_train_steps = 0

        for step, batch in enumerate(train_dataloader):
            # permute the tensor to go from shape (batch size, 3, max_tokens) to (3, batch size, max tokens)
            batch = batch.permute(1, 0, 2)

            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            batch_input_ids, batch_input_mask, batch_labels = batch

            # forward pass
            loss = model(batch_input_ids, token_type_ids=None,
                         attention_mask=batch_input_mask,
                         labels=batch_labels)

            # backward pass
            loss.backward()

            # track train loss
            train_loss += loss.item()
            nb_train_steps += 1

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

            # update parameters
            optimizer.step()
            model.zero_grad()

        # print train loss per epoch
        if not eval_f1_score_only:
            logging.info("Train loss: {}".format(train_loss/nb_train_steps))

        if valid_dataloader:
            model.eval()

            true_label_ids = []
            predictions_ids = []

            eval_loss = 0
            eval_accuracy = 0
            nb_eval_steps = 0

            for batch in valid_dataloader:
                batch = batch.permute(1, 0, 2)

                batch = tuple(t.to(device) for t in batch)
                batch_input_ids, batch_input_mask, batch_labels = batch

                with torch.no_grad():
                    logits = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = batch_labels.to('cpu').numpy()

                predictions_ids.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_label_ids.append(label_ids)

                if not eval_f1_score_only:
                    # compute loss and accuracy for this batch
                    with torch.no_grad():
                        tmp_eval_loss = model(batch_input_ids, token_type_ids=None,
                                              attention_mask=batch_input_mask,
                                              labels=batch_labels)

                        eval_loss += tmp_eval_loss.mean().item()

                        tmp_eval_accuracy = DimensionBertNer.flat_accuracy(logits, label_ids)
                        eval_accuracy += tmp_eval_accuracy

                        nb_eval_steps += 1

            # compute f1 score
            pred_tags, valid_tags = get_labels(predictions_ids, true_label_ids)
            score = f1_score(pred_tags, valid_tags)

            # save model weights if f1 score meets minimum and set a new standard
            if save_filename and score >= best_f1_score and score > save_min_f1_score:
                torch.save(model.state_dict(), save_filename)

                best_f1_score = score
                logging.info(f"\tF1-Score: {score :.5f} \t(model saved)")
            else:
                logging.info(f"\tF1-Score: {score :.5f}")

            # on last epoch, print a small report and the list of mislabeled evaluation samples
            if ep == nb_epochs-1:
                logging.info(classification_report(pred_tags, valid_tags))

                if valid_dataset:
                    print_mislabeled_samples(valid_dataset, pred_tags)

            # compute validation loss and accuracy, if requested
            # if not eval_f1_score_only:
            #     eval_loss = eval_loss / nb_eval_steps
            #
            #     logging.info("Validation loss: {}".format(eval_loss))
            #     logging.info("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

    logging.info(f"Best F1 score is {best_f1_score}.")


def start(dim_ner, valid_dataset, train_dataloader, valid_dataloader):
    if not os.path.exists('models'):
        os.makedirs('models')

    #model, optimizer = setup_model_for_finetuning(dim_ner.model, learning_rate = 5e-6)
    model, optimizer = setup_model_for_finetuning(dim_ner.model, learning_rate = 3e-6)

    train(model, optimizer, train_dataloader, valid_dataloader,
          nb_epochs=100, valid_dataset=valid_dataset,
          save_filename='models/dimension_ner_bert.pt',
          save_min_f1_score=0.96,
          eval_f1_score_only=False)

    torch.save(model.state_dict(), 'models/dimension_ner_bert_last_epoch.pt')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')  # no prefix

    dim_ner = DimensionBertNer()

    # train_batch_size is an important hyper params in fine tuning this model
    _, valid_dataset, train_dataloader, valid_dataloader = get_data_loaders(dim_ner.tokenizer,
                                                          "data/training_set.txt",
                                                          "data/validation_set.txt",
                                                          train_batch_size=10)

    #train_full_only(dim_ner, valid_dataset, train_dataloader, valid_dataloader)
    start(dim_ner, valid_dataset, train_dataloader, valid_dataloader)