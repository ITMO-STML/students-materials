import torch.nn as nn
import os
import sys
from pathlib import Path
import numpy as np
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

import torch
import random
from matplotlib import pyplot as plt
from typing import Dict, Any, List

import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_tools import (
        _compute_predictions,
        _accuracy_ordered,
        _accuracy_unordered,
        _accuracy_centre_flexible,
        _accuracy_ordered_flex_centre,
        _accuracy_centre,
        _accuracy_start,
        _accuracy_centre_flex,
        _accuracy_end,
        _compute_map,
        _plot_pr_curves,
        _plot_confusion_matrices,
        compute_error_rate
)

def evaluate(model, data_loader, loss_fn, file_name=None, phoneme_list=None, device='cuda', show_plot=True):
    model.eval()

    results = _compute_predictions(model, data_loader, loss_fn, device)


    if show_plot:
        metrics = {
            'loss': results['avg_loss'],
            'accuracy_ordered': _accuracy_ordered(results['true_labels'], results['pred_labels']),
            'accuracy_unordered': _accuracy_unordered(results['true_labels'], results['pred_labels']),
            'accuracy_ordered_flex_centre': _accuracy_ordered_flex_centre(results['true_labels'], results['pred_labels']),
            'accuracy_centre_flexible': _accuracy_centre_flexible(results['true_labels'], results['pred_labels']),

            'accuracy_start': _accuracy_start(results['true_labels'], results['pred_labels']),
            'accuracy_centre': _accuracy_centre(results['true_labels'], results['pred_labels']),
            'accuracy_centre_flex': _accuracy_centre_flex(results['true_labels'], results['pred_labels']),
            'accuracy_end': _accuracy_end(results['true_labels'], results['pred_labels']),

            'mean_average_precision_centre': _compute_map(results['true_labels'], results['scores_centre'], phoneme_list),
            'confusion_matrices': _plot_confusion_matrices(
                results['true_labels'], results['pred_labels'], phoneme_list, file_name
            )
        }
        _plot_pr_curves(results['scores_centre'], results['true_labels']['centre'], phoneme_list, file_name)
    else:
        metrics = {
            'loss': results['avg_loss'],
            'accuracy_ordered': _accuracy_ordered(results['true_labels'], results['pred_labels']),
            'accuracy_unordered': _accuracy_unordered(results['true_labels'], results['pred_labels']),
            'accuracy_ordered_flex_centre': _accuracy_ordered_flex_centre(results['true_labels'],
                                                                          results['pred_labels']),
            'accuracy_centre_flexible': _accuracy_centre_flexible(results['true_labels'], results['pred_labels']),

            'accuracy_start': _accuracy_start(results['true_labels'], results['pred_labels']),
            'accuracy_centre': _accuracy_centre(results['true_labels'], results['pred_labels']),
            'accuracy_centre_flex': _accuracy_centre_flex(results['true_labels'], results['pred_labels']),
            'accuracy_end': _accuracy_end(results['true_labels'], results['pred_labels']),

            'mean_average_precision_centre': _compute_map(results['true_labels'], results['scores_centre'],
                                                          phoneme_list)
        }

    return metrics


def train(model, train_loader, test_loader,  loss_fn, optimizer, scheduler, num_epochs, device,phoneme_list, writer=None, loss_info=False):
    model.train()
    loss_list = []
    test_loss_list = []
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        test_loss = 0.0
        counter_train = 0
        counter_test = 0

        for embs, labels, _ in tqdm(train_loader):
            counter_train += 1
            torch.cuda.empty_cache()
            embs = embs.to(device)
            labels = labels.to(device)

            start, centre, end = model(embs)

            loss_start = loss_fn(start, labels[:, 0])
            loss_centre = loss_fn(centre, labels[:, 1])
            loss_end = loss_fn(end, labels[:, 2])

            loss = loss_start + loss_centre + loss_end

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_train_loss = epoch_loss / counter_train
        loss_list.append(epoch_train_loss)

        # for embs, labels, _ in test_loader:
        #     counter_test += 1
        #     metrics = evaluate(model, test_loader, loss_fn, file_name=None, phoneme_list=phoneme_list, device=device, show_plot=False)
        #     test_loss += metrics['loss']
        # test_loss_list.append(test_loss / counter_test)
        metrics = evaluate(model, test_loader, loss_fn,
                           file_name=None,
                           phoneme_list=phoneme_list,
                           device=device,
                           show_plot=False)
        epoch_test_loss = metrics['loss']
        test_loss_list.append(epoch_test_loss)

        if writer is not None:
            writer.add_scalar("Loss/train", epoch_train_loss, epoch)  # [web:8]
            writer.add_scalar("Loss/test", epoch_test_loss, epoch)  # [web:8]

        scheduler.step()
    if loss_info:
        #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
        plt.plot(loss_list)
        plt.plot(test_loss_list)
        plt.show()

def test(model, test_loader, loss_fn, file_name, phoneme_list=None, device='cuda'):
    result = evaluate(model, test_loader, loss_fn, file_name=file_name, phoneme_list=phoneme_list, device=device, show_plot=True)
    print(f"Test Loss: {result['loss']:.4f}")
    print(f"Ordered Accuracy: {result['accuracy_ordered']:.4f}")
    print(f"Unordered Accuracy: {result['accuracy_unordered']:.4f}")
    print("Confusion Matrix - START")
    print(result['confusion_matrices']['start'])
    print("Confusion Matrix - CENTRE")
    print(result['confusion_matrices']['centre'])
    print("Confusion Matrix - END")
    print(result['confusion_matrices']['end'])
    return result

def get_phrase_prediciton(phrase, model, phoneme_list, device):

    prob_distributions = {'start': [], 'centre': [], 'end': []}
    y_pred = {'start': [], 'centre': [], 'end': []}
    result = []

    with torch.no_grad():
        start = []
        centre = []
        end = []
        for embs in phrase:
            embs = embs.unsqueeze(0).to(device)
            start_out, centre_out, end_out = model(embs)

            # Apply softmax to get probabilities
            start_probs = torch.softmax(start_out, dim=1)
            centre_probs = torch.softmax(centre_out, dim=1)
            end_probs = torch.softmax(end_out, dim=1)

            # Get the probability distributions
            start_dist = {phoneme: start_probs[0][i].item() for i, phoneme in enumerate(phoneme_list)}
            start_dist = sorted(start_dist.items(), key=lambda x: x[1], reverse=True)
            centre_dist = {phoneme: centre_probs[0][i].item() for i, phoneme in enumerate(phoneme_list)}
            centre_dist = sorted(centre_dist.items(), key=lambda x: x[1], reverse=True)
            end_dist = {phoneme: end_probs[0][i].item() for i, phoneme in enumerate(phoneme_list)}
            end_dist = sorted(end_dist.items(), key=lambda x: x[1], reverse=True)

            # Store the distributions
            prob_distributions['start'].append(start_dist)
            prob_distributions['centre'].append(centre_dist)
            prob_distributions['end'].append(end_dist)

            # Get the predicted classes (original functionality)
            pred_start = torch.argmax(start_out, dim=1)
            start.append(phoneme_list[pred_start[0]])
            pred_centre = torch.argmax(centre_out, dim=1)
            centre.append(phoneme_list[pred_centre[0]])
            pred_end = torch.argmax(end_out, dim=1)
            end.append(phoneme_list[pred_end[0]])

            preds = []
            preds.append(pred_start.cpu().numpy().tolist())
            preds.append(pred_centre.cpu().numpy().tolist())
            preds.append(pred_end.cpu().numpy().tolist())
            res = [phoneme_list[preds[i][0]] for i in range(len(preds))]
            res.append('')
            result += res

        y_pred['start'] = start
        y_pred['centre'] = centre
        y_pred['end'] = end

    return y_pred, prob_distributions, result
