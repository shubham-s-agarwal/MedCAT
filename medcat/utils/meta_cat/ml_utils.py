import os
import random
import math
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from typing import List, Optional, Tuple, Any, Dict
from torch import nn
from scipy.special import softmax
import matplotlib.pyplot as plt
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.tokenizers_med.meta_cat_tokenizers import TokenizerWrapperBase
from sklearn.metrics import classification_report, precision_recall_fscore_support,f1_score, confusion_matrix, accuracy_score,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.utils import class_weight
# from medcat.utils.meta_cat.meta_cat_loss import DiceLoss,FocalLoss

import logging

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_batch_piped_data(data: List, start_ind: int, end_ind: int, device: torch.device, pad_id: int) -> Tuple:
    """Creates a batch given data and start/end that denote batch size, will also add
    padding and move to the right device.

    Args:
        data (List[List[int], int, Optional[int]]):
            Data in the format: [[<[input_ids]>, <cpos>, Optional[int]], ...], the third column is optional
            and represents the output label
        start_ind (int):
            Start index of this batch
        end_ind (int):
            End index of this batch
        device (torch.device):
            Where to move the data
        pad_id (int):
            Padding index

    Returns:
        x ():
            Same as data, but subsetted and as a tensor
        cpos ():
            Center positions for the data
    """
    max_seq_len = max([len(x[0]) for x in data])
    x = [x[0][0:max_seq_len] + [pad_id]*max(0, max_seq_len - len(x[0])) for x in data[start_ind:end_ind]]
    cpos = [x[1] for x in data[start_ind:end_ind]]
    y = None
    if len(data[0]) == 3:
        # Means we have the y column
        y = torch.tensor([x[2] for x in data[start_ind:end_ind]], dtype=torch.long).to(device)

    x = torch.tensor(x, dtype=torch.long).to(device)
    cpos = torch.tensor(cpos, dtype=torch.long).to(device)

    attention_masks = (x != 0).type(torch.int)

    return x, cpos,attention_masks,y


def predict(model: nn.Module, data: List, config: ConfigMetaCAT) -> Tuple:
    """Predict on data used in the meta_cat.pipe

    Args:
        data (List[List[List[int], int]]):
            Data in the format: [[<input_ids>, <cpos>], ...]
        config (medcat.config_meta_cat.ConfigMetaCAT):
            Configuration for this meta_cat instance.

    Returns:
        predictions (List[int]):
            For each row of input data a prediction
        confidence (List[float]):
            For each prediction a confidence value
    """

    pad_id = config.model['padding_idx']
    batch_size = config.general['batch_size_eval']
    device = config.general['device']
    ignore_cpos = config.model['ignore_cpos']

    model.eval()
    model.to(device)

    num_batches = math.ceil(len(data) / batch_size)
    all_logits = []

    with torch.no_grad():
        for i in range(num_batches):
            x, cpos,attention_masks, _ = create_batch_piped_data(data, i*batch_size, (i+1)*batch_size, device=device, pad_id=pad_id)
            logits = model(x, cpos,attention_mask=attention_masks, ignore_cpos=ignore_cpos)
            all_logits.append(logits.detach().cpu().numpy())

    predictions = []
    confidences = []

    # Can be that there are not logits, data is empty
    if all_logits:
        logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(logits, axis=1)
        confidences = np.max(softmax(logits, axis=1), axis=1)

    return predictions, confidences


def split_list_train_test(data: List, test_size: int, shuffle: bool = True) -> Tuple:
    """Shuffle and randomly split data

    Args:
        data
        test_size
        shuffle
    """
    if shuffle:
        random.shuffle(data)

    print("This is the data",data)

    X_features = [x[:-1] for x in data]
    print("X_features", X_features)
    y_labels = [x[-1] for x in data]

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=test_size, stratify=y_labels,
                                                        random_state=42)

    train_data = [x + [y] for x,y in zip(X_train, y_train)]
    test_data = [x + [y] for x, y in zip(X_test, y_test)]

    print("train_data", train_data)

    # test_ind = int(len(data) * test_size)
    # test_data = data[:test_ind]
    # train_data = data[test_ind:]

    return train_data, test_data


def print_report(epoch: int, running_loss: List, all_logits: List, y: Any, name: str = 'Train') -> None:
    """Prints some basic stats during training

    Args:
        epoch
        running_loss
        all_logits
        y
        name
    """
    if all_logits:
        logger.info('Epoch: %d %s %s', epoch, "*"*50, name)
        #logger.info(classification_report(y, np.argmax(np.concatenate(all_logits, axis=0), axis=1)))
        print('Epoch: ', epoch,name)
        print("Accuracy: ",accuracy_score(y, np.argmax(np.concatenate(all_logits, axis=0), axis=1)))
        print("F1-score: ",f1_score(y, np.argmax(np.concatenate(all_logits, axis=0), axis=1),average='macro'))


#
# def multi_class_recall_specificity(y_true, y_pred, recall_weight, spec_weight):
#     print("y_pred",y_pred)
#     _report = classification_report(y_true, y_pred,
#                                     output_dict=True)
#     cnf_matrix = confusion_matrix(y_true, y_pred)
#     FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
#     FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
#     TP = np.diag(cnf_matrix)
#     TN = cnf_matrix.sum() - (FP + FN + TP)
#
#     FP = FP.astype(float)
#     # FN = FN.astype(float)
#     # TP = TP.astype(float)
#     TN = TN.astype(float)
#
#     specificity = TN / (TN + FP)
#
#     # Calculate the final loss
#     loss = 1.0 - (recall_weight * _report['macro avg']['recall'] + spec_weight * specificity)
#
#     return loss.item()  # Return as a Python float
#
# def custom_loss(recall_weight, spec_weight):
#     def recall_spec_loss(y_true, y_pred):
#         return multi_class_recall_specificity(y_true, y_pred, recall_weight, spec_weight)
#
#     # Returns the (y_true, y_pred) loss function
#     return recall_spec_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss

def train_model(model: nn.Module, data: List, config: ConfigMetaCAT, save_dir_path: Optional[str] = None,model_arch_config=None) -> Dict:
    """Trains a LSTM model (for now) with autocheckpoints

    Args:
        data
        config
        save_dir_path
    """
    # Get train/test from data
    train_data, test_data = split_list_train_test(data, test_size=config.train['test_size'], shuffle=config.train['shuffle_data'])
    device = torch.device(config.general['device']) # Create a torch device

    class_weights = config.train['class_weights']
    print("WEIGHTS:",class_weights)
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        if config.train['loss_funct'] == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(weight=class_weights) # Set the criterion to Cross Entropy Loss
        elif config.train['loss_funct'] == 'focal_loss':
            criterion = FocalLoss(alpha=class_weights, gamma=config.train['gamma'])
        # criterion = nn.BCEWithLogitsLoss()
    else:
        if config.train['loss_funct'] == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()  # Set the criterion to Cross Entropy Loss
        elif config.train['loss_funct'] == 'focal_loss':
            criterion = FocalLoss(gamma=config.train['gamma'])

        # criterion = SelfAdjDiceLoss()
        # criterion = DiceLoss(with_logits=True, smooth=1, ohem_ratio=0,
        #                         alpha=0.01,
        #                         index_label_position=True, reduction="mean")

        # criterion = FocalLoss(gamma=4, reduction="mean")

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    def initialize_model(model,data,batch_size,lr,epochs=4):
        """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """

        # Instantiate Bert Classifier
        bert_classifier = model

        # Create the optimizer
        optimizer = AdamW(bert_classifier.parameters(),
                          lr=lr,  # Default learning rate
                          eps=1e-8, # Default epsilon value
                          weight_decay = 1e-5
                          )

        # Total number of training steps
        total_steps = int((len(data)/batch_size) * epochs)
        print('Total steps: {}'.format(total_steps))

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=total_steps)
        return bert_classifier, optimizer, scheduler


    batch_size = config.train['batch_size']
    batch_size_eval = config.general['batch_size_eval']
    pad_id = config.model['padding_idx']
    nepochs = config.train['nepochs']
    ignore_cpos = config.model['ignore_cpos']
    num_batches = math.ceil(len(train_data) / batch_size)
    num_batches_test = math.ceil(len(test_data) / batch_size_eval)
    optimizer = optim.Adam(parameters, lr=config.train['lr'],weight_decay = 1e-5)
    if model_arch_config is not None:
        if model_arch_config['lr_scheduler'] is True:
            model, optimizer, scheduler = initialize_model(model,train_data,batch_size,config.train['lr'],epochs=nepochs)

    model.to(device)  # Move the model to device

    # Can be pre-calculated for the whole dataset
    y_test = [x[2] for x in test_data]
    y_train = [x[2] for x in train_data]
    print("Y_train - ",set(y_train))
    winner_report: Dict = {}
    for epoch in range(nepochs):
        running_loss = []
        all_logits = []
        model.train()
        for i in range(num_batches):
            x, cpos,attention_masks, y = create_batch_piped_data(train_data, i*batch_size, (i+1)*batch_size, device=device, pad_id=pad_id)
            logits = model(x, attention_mask = attention_masks, center_positions=cpos,model_arch_config=model_arch_config)
            # print("Y",y)
            loss = criterion(logits, y)

            loss.backward()
            # Track loss and logits
            running_loss.append(loss.item())
            all_logits.append(logits.detach().cpu().numpy())

            parameters = filter(lambda p: p.requires_grad, model.parameters())
            nn.utils.clip_grad_norm_(parameters, 0.15)
            optimizer.step()
            if model_arch_config is not None:
                if model_arch_config['lr_scheduler'] is True:
                    scheduler.step()
            # current_lr = optimizer.param_groups[0]['lr']
            # print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")

        all_logits_test = []
        running_loss_test = []
        model.eval()

        with torch.no_grad():
            for i in range(num_batches_test):
                x, cpos,attention_masks, y = create_batch_piped_data(test_data, i*batch_size_eval, (i+1)*batch_size_eval, device=device, pad_id=pad_id)

                logits = model(x, attention_mask = attention_masks, center_positions=cpos,model_arch_config=model_arch_config)

                # Track loss and logits
                running_loss_test.append(loss.item())
                all_logits_test.append(logits.detach().cpu().numpy())

        print_report(epoch, running_loss, all_logits, y=y_train, name='Train')
        print_report(epoch, running_loss_test, all_logits_test, y=y_test, name='Test')

        _report = classification_report(y_test, np.argmax(np.concatenate(all_logits_test, axis=0), axis=1), output_dict=True)

        if not winner_report or _report[config.train['metric']['base']][config.train['metric']['score']] > \
                winner_report['report'][config.train['metric']['base']][config.train['metric']['score']]:

            report = classification_report(y_test, np.argmax(np.concatenate(all_logits_test, axis=0), axis=1), output_dict=True)
            cm = confusion_matrix(y_test, np.argmax(np.concatenate(all_logits_test, axis=0), axis=1),normalize='true')
            report_train = classification_report(y_train, np.argmax(np.concatenate(all_logits, axis=0), axis=1), output_dict=True)

            winner_report['confusion_matrix'] = cm
            winner_report['report'] = report
            winner_report['report_train'] = report_train
            winner_report['epoch'] = epoch

            # Save if needed
            if config.train['auto_save_model']:
                if save_dir_path is None:
                    raise Exception(
                        "The `save_dir_path` argument is required if `aut_save_model` is `True` in the config")
                else:
                    path = os.path.join(save_dir_path, 'model.dat')
                    torch.save(model.state_dict(), path)
                    logger.info("\n##### Model saved to %s at epoch: %d and %s/%s: %s #####\n", path, epoch, config.train['metric']['base'],
                          config.train['metric']['score'], winner_report['report'][config.train['metric']['base']][config.train['metric']['score']])

    return winner_report

def eval_model(model: nn.Module, data: List, config: ConfigMetaCAT, tokenizer: TokenizerWrapperBase) -> Dict:
    """Evaluate a trained model on the provided data

    Args:
        model
        data
        config
    """
    device = torch.device(config.general['device']) # Create a torch device
    batch_size_eval = config.general['batch_size_eval']
    pad_id = config.model['padding_idx']
    ignore_cpos = config.model['ignore_cpos']
    class_weights = config.train['class_weights']

    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights) # Set the criterion to Cross Entropy Loss
    else:
        criterion = nn.CrossEntropyLoss() # Set the criterion to Cross Entropy Loss

    y_eval = [x[2] for x in data]
    num_batches = math.ceil(len(data) / batch_size_eval)
    running_loss = []
    all_logits = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(num_batches):
            x, cpos,attention_masks, y = create_batch_piped_data(data, i*batch_size_eval, (i+1)*batch_size_eval, device=device, pad_id=pad_id)
            logits = model(x, cpos, ignore_cpos=ignore_cpos)
            loss = criterion(logits, y)

            # Track loss and logits
            running_loss.append(loss.item())
            all_logits.append(logits.detach().cpu().numpy())

    print_report(0, running_loss, all_logits, y=y_eval, name='Eval')

    score_average = config.train['score_average']
    predictions = np.argmax(np.concatenate(all_logits, axis=0), axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y_eval, predictions, average=score_average)

    labels = [name for (name, _) in sorted(config.general['category_value2id'].items(), key=lambda x:x[1])]
    confusion = pd.DataFrame(
        data=confusion_matrix(y_eval, predictions,),
        columns=["true " + label for label in labels],
        index=["predicted " + label for label in labels],
    )


    examples: Dict = {'FP': {}, 'FN': {}, 'TP': {}}
    id2category_value = {v: k for k, v in config.general['category_value2id'].items()}
    for i, p in enumerate(predictions):
        y = id2category_value[y_eval[i]]
        p = id2category_value[p]
        c = data[i][1]
        tkns = data[i][0]
        assert tokenizer.hf_tokenizers is not None
        text = tokenizer.hf_tokenizers.decode(tkns[0:c]) + " <<"+ tokenizer.hf_tokenizers.decode(tkns[c:c+1]).strip() + ">> " + \
            tokenizer.hf_tokenizers.decode(tkns[c+1:])
        info = "Predicted: {}, True: {}".format(p, y)
        if p != y:
            # We made a mistake
            examples['FN'][y] = examples['FN'].get(y, []) + [(info, text)]
            examples['FP'][p] = examples['FP'].get(p, []) + [(info, text)]
        else:
            examples['TP'][y] = examples['TP'].get(y, []) + [(info, text)]

    return {'precision': precision, 'recall': recall, 'f1': f1, 'examples': examples, 'confusion matrix': confusion}
