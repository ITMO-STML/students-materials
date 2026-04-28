import torch
from Levenshtein import distance, editops


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve,average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import permutations
from Decoding_tools import viterbi_decode, viterbi_lookahead, write_to_seg
from project_paths import TABLES_DIR, ensure_project_dirs


def _compute_predictions(model, data_loader, loss_fn, device):

    total_loss = 0.0

    true_labels = {'start': [], 'centre': [], 'end': [], 'centre_flex': []}
    pred_labels = {'start': [], 'centre': [], 'end': [], 'centre_flex': []}
    scores_centre = []

    raw_labels = []
    raw_preds = []

    with torch.no_grad():
        for embs, labels, _ in data_loader:
            embs, labels = embs.to(device), labels.to(device)
            start_out, centre_out, end_out = model(embs)

            loss = sum(loss_fn(out, labels[:, i]) for i, out in enumerate([start_out, centre_out, end_out]))
            total_loss += loss.item()

            pred_start = torch.argmax(start_out, dim=1)
            pred_centre = torch.argmax(centre_out, dim=1)
            pred_end = torch.argmax(end_out, dim=1)

            probs_centre = torch.softmax(centre_out, dim=1)

            true_labels['start'].extend(labels[:, 0].cpu().numpy())
            true_labels['centre'].extend(labels[:, 1].cpu().numpy())
            true_labels['end'].extend(labels[:, 2].cpu().numpy())
            pred_labels['start'].extend(pred_start.cpu().numpy())
            pred_labels['centre'].extend(pred_centre.cpu().numpy())
            pred_labels['end'].extend(pred_end.cpu().numpy())
            scores_centre.extend(probs_centre.cpu().numpy())

            # For flexible centre match
            for i in range(len(labels)):
                true_set = set(labels[i].tolist())
                pred_c = pred_centre[i].item()
                true_c = labels[i, 1].item()
                pred_labels['centre_flex'].append(pred_c)
                true_labels['centre_flex'].append(true_c)

                raw_labels.append(labels[i].tolist())
                raw_preds.append([pred_start[i].item(), pred_centre[i].item(), pred_end[i].item()])


    return {
        'avg_loss': total_loss / len(data_loader),
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'scores_centre': scores_centre,
        'raw_labels': raw_labels,
        'raw_preds': raw_preds
    }


def _accuracy_ordered(y_true, y_pred):
  correct = sum(
      ps == ts and pc == tc and pe == te
      for ps, pc, pe, ts, tc, te in zip(
          y_pred['start'], y_pred['centre'], y_pred['end'],
          y_true['start'], y_true['centre'], y_true['end']
      )
  )
  return correct / len(y_true['centre'])


def _accuracy_unordered(y_true, y_pred):
    total_correct = 0
    total_elements = len(y_true['start'])

    for true_seq, pred_seq in zip(zip(*y_true.values()), zip(*y_pred.values())):
        #print(true_seq, pred_seq)
        #print(tuple(pred_seq), set(permutations(true_seq)))
        # Проверяем, что множества элементов совпадают (одинаковые элементы)
        if set(true_seq) != set(pred_seq):
            continue

        # Проверяем, является ли pred_seq перестановкой true_seq

        if tuple(pred_seq) in set(permutations(true_seq)):
            total_correct += 1

    return total_correct / total_elements if total_elements > 0 else 0.0


def _accuracy_centre_flexible(y_true, y_pred):
    correct = sum(p in {t_s, t_c, t_e} for p, t_s, t_c, t_e in zip(
        y_pred['centre'], y_true['start'], y_true['centre'], y_true['end']
    ))
    return correct / len(y_true['centre'])


def _accuracy_ordered_flex_centre(y_true, y_pred):
    correct = sum(
        (ps == ts and pe == te and (pc == ts or pc == te))
        for ps, pc, pe, ts, tc, te in zip(
            y_pred['start'], y_pred['centre'], y_pred['end'],
            y_true['start'], y_true['centre'], y_true['end']
        )
    )
    return correct / len(y_true['centre'])


def _accuracy_start(y_true, y_pred):
    correct = sum(
      ps == ts for ps, ts in zip(
          y_pred['start'],
          y_true['start'],
          )
      )
    return correct / len(y_true['centre'])


def _accuracy_centre(y_true, y_pred):
    correct = sum(
      pc == tc for pc, tc in zip(
          y_pred['centre'],
          y_true['centre'],
          )
      )
    return correct / len(y_true['centre'])


def _accuracy_centre_flex(y_true, y_pred):
    correct = sum(
        (ps == ts and te == pe) and ((pc == ts or pe == te) or (pc == tc))
        for ps, pc, pe, ts, tc, te in zip(
            y_pred['start'], y_pred['centre'], y_pred['end'],
            y_true['start'], y_true['centre'], y_true['end']
        )
    )
    return correct / len(y_true['centre'])


def _accuracy_end(y_true, y_pred):
    correct = sum(
      pe == te for pe, te in zip(
          y_pred['end'],
          y_true['end'],
          )
      )
    return correct / len(y_true['centre'])


def _compute_map(y_true, scores, phoneme_list):
    num_classes = len(phoneme_list)
    y_true_bin = label_binarize(y_true['centre'], classes=list(range(num_classes)))
    return average_precision_score(y_true_bin, scores, average='macro')


def _plot_pr_curves(scores, y_true_centre, phoneme_list, file_name):
    ensure_project_dirs()
    num_classes = len(phoneme_list)
    y_true_bin = label_binarize(y_true_centre, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(15, 13))
    colors = plt.cm.get_cmap('tab10', num_classes)

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], [s[i] for s in scores])
        ap = average_precision_score(y_true_bin[:, i], [s[i] for s in scores])
        label = f"{phoneme_list[i]} (AP = {ap:.2f})" if phoneme_list else f"Class {i}"
        ax.plot(recall, precision, label=label, color=colors(i))

    ax.set_title("Precision-Recall Curves (Centre Position)", fontsize=20)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc='best', fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(TABLES_DIR / f"prc_{file_name}.png")


def _plot_confusion_matrices(y_true, y_pred, phoneme_list, file_name):
    ensure_project_dirs()
    fig, axs = plt.subplots(1, 4, figsize=(30, 10))
    fig.suptitle("Confusion Matrices", fontsize=36)

    matrices = {}
    for i, key in enumerate(['start', 'centre', 'end', 'centre_flex']):
        conf = confusion_matrix(y_true[key], y_pred[key], normalize='true')
        cmap = ['Blues', 'Greens', 'Oranges', 'Purples'][i]

        disp = ConfusionMatrixDisplay(conf, display_labels=phoneme_list)
        disp.plot(ax=axs[i], cmap=cmap, colorbar=False, xticks_rotation=45)

        axs[i].set_title(f"{key.upper()} position", fontsize=28)
        axs[i].set_xlabel('Predicted label', fontsize=22)
        axs[i].set_ylabel('True label', fontsize=22)

        axs[i].tick_params(axis='both', which='major', labelsize=18)

        matrices[key] = conf

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make space for big suptitle
    plt.savefig(TABLES_DIR / f"cm_{file_name}.png")
    return matrices


def cer(true, pred, ignore_stress=True):
    tr = []
    pr = []
    if ignore_stress:
        for t in true:
            if t[0] in 'aeouyi':
                tr.append(t[0])
            else:
                tr.append(t)
        for p in pred:
            if p[0] in 'aeouyi':
                pr.append(p[0])
            else:
                pr.append(p)

    lev_dist = distance(tr, pr)
    cer_score = lev_dist / len(true) if true else 0.0
    return lev_dist, cer_score

def pfer(true, pred):
    return 0

def normalize_prob_dist(prob_dist, eps=1e-8):
    out = []
    for frame in prob_dist:
        frame_dict = {ph: max(p, eps) for ph, p in frame}
        out.append(frame_dict)
    return out




def phonemes_with_ends(starts, labels, total_duration):
    result = []
    for i, (s, lab) in enumerate(zip(starts, labels)):
        e = starts[i + 1] if i + 1 < len(starts) else total_duration
        result.append((s, e, lab))
    return result


def compute_error_rate(prob_distributions, true_labelling_list, ignore_stress=True, la_penalty=3.0, la_windows=2):

    probability_dist = {
            "start": normalize_prob_dist(prob_distributions["start"]),
            "center": normalize_prob_dist(prob_distributions["centre"]),
            "end": normalize_prob_dist(prob_distributions["end"]),
        }

    phonemes, phonemes_no_rep, boundaries = viterbi_decode(probability_dist)
    phonemes_la, phonemes_no_rep_la, boundaries_la = viterbi_lookahead(probability_dist,
                                                                       lambda_penalty=la_penalty,
                                                                       lookahead=la_windows)

    basic_levinstein, basic_cer = cer(true_labelling_list, phonemes_no_rep, ignore_stress=ignore_stress)
    la_levinstein, la_cer = cer(true_labelling_list, phonemes_no_rep_la, ignore_stress=ignore_stress)

    basic_pfer = pfer(true_labelling_list, phonemes_no_rep)
    la_pfer = pfer(true_labelling_list, phonemes_no_rep_la)

    return {
        'probability_distribution': probability_dist,
        'viterbi_decoded_phonemes': phonemes,
        'viterbi_decoded_phonemes_no_repetition': phonemes_no_rep,
        'viterbi_decoded_phonemes_la': phonemes_la,
        'viterbi_decoded_phonemes_no_repetition_la': phonemes_la,
        'metrics': {
            'basic_levinstein': basic_levinstein,
            'basic_cer': basic_cer,
            'la_levinstein': la_levinstein,
            'la_cer': la_cer,
            'basic_pfer': basic_pfer,
            'la_pfer': la_pfer,
        }
    }


def cer_detailed(true_seq, pred_seq):
    """
    true_seq: list[str]
    pred_seq: list[str]
    """
    if len(true_seq) == 0:
        return 0.0

    dist = distance(true_seq, pred_seq)
    ops = editops(true_seq, pred_seq)

    subs = sum(1 for op in ops if op[0] == "replace")
    ins = sum(1 for op in ops if op[0] == "insert")
    del_ = sum(1 for op in ops if op[0] == "delete")

    return dist / len(true_seq), subs, ins, del_

