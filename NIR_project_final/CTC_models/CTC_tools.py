import os
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter
from tqdm import tqdm
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

def load_selected_embeddings_ctc(
        root_dir,
        phoneme_list,
        speakers,
        max_sequences_per_speaker=None
):
    info = []
    selected_embeddings = []
    selected_phonemes = []

    for speaker in speakers:
        speaker_root_dir = os.path.join(root_dir, speaker)
        speaker_counter = Counter()
        speaker_seq_count = 0

        for root, _, files in os.walk(speaker_root_dir):

            if max_sequences_per_speaker is not None and speaker_seq_count >= max_sequences_per_speaker:
                break

            for file in files:
                if max_sequences_per_speaker is not None and speaker_seq_count >= max_sequences_per_speaker:
                    break

                if not file.endswith("_no_av_ph.txt"):
                    continue

                name = file.replace("_no_av_ph.txt", "")
                phoneme_file_path = os.path.join(root, file)
                emb_file_path = os.path.join(root, name + "_no_av_embs.npy")

                if not os.path.exists(emb_file_path):
                    continue

                with open(phoneme_file_path, "r", encoding="utf-8") as f:
                    phonemes = [line.strip() for line in f.readlines()]

                emb_array = np.load(emb_file_path)

                if len(phonemes) != len(emb_array):
                    continue

                emb_sequence = []
                phoneme_sequence = []

                for phoneme, emb in zip(phonemes, emb_array):
                    parts = phoneme.split('_')
                    if len(parts) != 3:
                        continue

                    base_ph = parts[1]
                    if base_ph not in phoneme_list:
                        continue

                    emb_sequence.append(emb)
                    phoneme_sequence.append(base_ph)
                    speaker_counter[base_ph] += 1

                if len(phoneme_sequence) > 0:
                    selected_embeddings.append(
                        torch.tensor(np.stack(emb_sequence), dtype=torch.float32)
                    )
                    selected_phonemes.append(phoneme_sequence)
                    speaker_seq_count += 1

        info.append({
            speaker: {
                "num_sequences": speaker_seq_count,
                "phoneme_distribution": speaker_counter
            }
        })

        print(f"{speaker}: {speaker_seq_count} sequences")

    return selected_embeddings, selected_phonemes, info


class CTCPhonemeDataset(Dataset):
    def __init__(self, sequences, targets, phoneme_list):
        """
        sequences: list[Tensor[T, D]]
        targets:   list[list[str]]
        """
        self.sequences = sequences
        self.targets = targets

        # 0 — blank
        self.phoneme2idx = {ph: i + 1 for i, ph in enumerate(phoneme_list)}
        self.idx2phoneme = {i + 1: ph for i, ph in enumerate(phoneme_list)}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx]
        y_raw = self.targets[idx]

        y = [y_raw[0]]
        for i in range(1, len(y_raw)):
            if y_raw[i] != y_raw[i - 1]:
                y.append(y_raw[i])

        y_idx = torch.tensor(
            [self.phoneme2idx[p] for p in y],
            dtype=torch.long
        )

        return x, y_idx


def ctc_collate_fn(batch):
    xs, ys = zip(*batch)

    input_lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    target_lengths = torch.tensor([y.size(0) for y in ys], dtype=torch.long)

    # [B, T_max, D]
    xs_padded = pad_sequence(xs, batch_first=True)

    # targets ДОЛЖНЫ БЫТЬ 1D
    ys_concat = torch.cat(ys)

    return xs_padded, ys_concat, input_lengths, target_lengths


def attention_collate_fn(batch, pad_idx=0):
    """
    Collate function for the attention decoder.
    Returns padded phoneme targets so we can use teacher forcing.
    """
    xs, ys = zip(*batch)

    input_lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    target_lengths = torch.tensor([y.size(0) for y in ys], dtype=torch.long)

    xs_padded = pad_sequence(xs, batch_first=True)
    # целевые последовательности также представляем в виде матрицы тк аттеншну надо смотреть на пердыдущие фонемы
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=pad_idx)

    return xs_padded, ys_padded, input_lengths, target_lengths


def train_ctc(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        num_epochs,
        device,
        writer,
        save_path
):
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    model.to(device)

    best_test_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x, y, x_lens, y_lens in train_loader:
            x = x.to(device)
            y = y.to(device)
            x_lens = x_lens.to(device)
            y_lens = y_lens.to(device)

            logits = model(x)  # [B, T, C]
            log_probs = logits.log_softmax(-1)  # log-softmax
            log_probs = log_probs.transpose(0, 1)  # [T, B, C]

            loss = ctc_loss(log_probs, y, x_lens, y_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for x, y, x_lens, y_lens in test_loader:
                x = x.to(device)
                y = y.to(device)
                x_lens = x_lens.to(device)
                y_lens = y_lens.to(device)

                logits = model(x)
                log_probs = logits.log_softmax(-1).transpose(0, 1)
                loss = ctc_loss(log_probs, y, x_lens, y_lens)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.flush()

        print(
            f"Epoch {epoch + 1}: "
            f"train={train_loss:.4f} | test={test_loss:.4f}"
        )
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "train_loss": train_loss,
                    "test_loss": best_test_loss,
                },
                save_path,
            )
            print(f"  → Best model saved at epoch {epoch + 1} with test_loss={best_test_loss:.4f}")

    print(f"Training finished. Best test loss: {best_test_loss:.4f} at epoch {best_epoch + 1}.")


def ctc_greedy_decode(logits, input_lengths, idx2phoneme):
    """
    logits: [B, T, C]
    """
    preds = logits.argmax(-1)  # [B, T]
    results = []

    for seq, T in zip(preds, input_lengths):
        seq = seq[:T]

        decoded = []
        prev = 0  # blank

        for idx in seq.cpu().numpy():
            if idx != 0 and idx != prev:
                decoded.append(idx2phoneme[idx])
            prev = idx

        results.append(decoded)

    return results

def build_decoder_inputs (targets, start_idx=0):
    """
    Shift targets to the right for teacher forcing.
    If targets = [p1, p2, p3], decoder inputs become [<s>, p1, p2].
    """
    decoder_inputs = torch.zeros_like(targets)
    decoder_inputs[:, 0] = start_idx
    decoder_inputs[:, 1:] = targets[:, :-1]
    return decoder_inputs


def build_stop_targets(target_lengths, max_target_len, device):
    """
    Build stop-token targets for decoder steps.
    The final valid decoder step is marked with 1.0, earlier valid steps with 0.0.
    Padded positions after the sequence are also marked with 1.0 so the model learns to stop.
    """
    step_ids = torch.arange(max_target_len, device=device).unsqueeze(0)
    last_valid_step = (target_lengths - 1).unsqueeze(1)
    stop_targets = (step_ids >= last_valid_step).float()
    return stop_targets


def train_attention_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    writer,
    save_path,
    lambda_att=1.0,
    guided_loss_fn=None,
):
    if guided_loss_fn is None:
        guided_loss_fn = GuidedAttentionLoss(g=0.2)

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    stop_loss_fn = nn.BCEWithLogitsLoss()

    model.to(device)
    guided_loss_fn.to(device)

    best_test_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_total = 0.0
        train_ce = 0.0
        train_stop = 0.0
        train_att = 0.0

        for x, y, x_lens, y_lens in tqdm(train_loader, desc=f"train {epoch+1}/{num_epochs}"):
            x = x.to(device)
            y = y.to(device)
            x_lens = x_lens.to(device)
            y_lens = y_lens.to(device)

            decoder_inputs = build_decoder_inputs(y).to(device)
            stop_targets = build_stop_targets(y_lens, y.size(1), device)
            logits, stop_logits, attention, _ = model(x, decoder_inputs, x_lens)

            ce_loss = ce_loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            stop_loss = stop_loss_fn(stop_logits, stop_targets)
            att_loss = guided_loss_fn(attention, y_lens, x_lens)
            loss = ce_loss + stop_loss + lambda_att * att_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total += loss.item()
            train_ce += ce_loss.item()
            train_stop += stop_loss.item()
            train_att += att_loss.item()

        train_total /= len(train_loader)
        train_ce /= len(train_loader)
        train_stop /= len(train_loader)
        train_att /= len(train_loader)

        model.eval()
        test_total = 0.0
        test_ce = 0.0
        test_stop = 0.0
        test_att = 0.0

        with torch.no_grad():
            for x, y, x_lens, y_lens in tqdm(test_loader, desc=f"eval {epoch+1}/{num_epochs}"):
                x = x.to(device)
                y = y.to(device)
                x_lens = x_lens.to(device)
                y_lens = y_lens.to(device)

                decoder_inputs = build_decoder_inputs(y).to(device)
                stop_targets = build_stop_targets(y_lens, y.size(1), device)
                logits, stop_logits, attention, _ = model(x, decoder_inputs, x_lens)

                ce_loss = ce_loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                stop_loss = stop_loss_fn(stop_logits, stop_targets)
                att_loss = guided_loss_fn(attention, y_lens, x_lens)
                loss = ce_loss + stop_loss + lambda_att * att_loss

                test_total += loss.item()
                test_ce += ce_loss.item()
                test_stop += stop_loss.item()
                test_att += att_loss.item()

        test_total /= len(test_loader)
        test_ce /= len(test_loader)
        test_stop /= len(test_loader)
        test_att /= len(test_loader)

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("AttentionLoss/train_total", train_total, epoch)
        writer.add_scalar("AttentionLoss/train_ce", train_ce, epoch)
        writer.add_scalar("AttentionLoss/train_stop", train_stop, epoch)
        writer.add_scalar("AttentionLoss/train_guided", train_att, epoch)
        writer.add_scalar("AttentionLoss/test_total", test_total, epoch)
        writer.add_scalar("AttentionLoss/test_ce", test_ce, epoch)
        writer.add_scalar("AttentionLoss/test_stop", test_stop, epoch)
        writer.add_scalar("AttentionLoss/test_guided", test_att, epoch)
        writer.flush()

        print(
            f"Epoch {epoch+1}: "
            f"train_total={train_total:.4f} | train_ce={train_ce:.4f} | train_stop={train_stop:.4f} | train_att={train_att:.4f} | "
            f"test_total={test_total:.4f} | test_ce={test_ce:.4f} | test_stop={test_stop:.4f} | test_att={test_att:.4f}"
        )

        if test_total < best_test_loss:
            best_test_loss = test_total
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "test_total": best_test_loss,
                    "lambda_att": lambda_att,
                },
                save_path,
            )
            print(f"  -> Best attention model saved at epoch {epoch+1} with test_total={best_test_loss:.4f}")

    print(f"Training finished. Best test loss: {best_test_loss:.4f} at epoch {best_epoch+1}.")


class CTCModel_linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_phonemes, num_layers=4, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # +1 для blank
        self.ctc_head = nn.Linear(64, num_phonemes + 1)

    def forward(self, x):
        # x: [B, T, D]
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.dropout(x)
        x = self.relu(self.linear4(x))
        x = self.dropout(x)
        logits = self.ctc_head(x)  # [B, T, V+1]
        return logits


class CTCModel_lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_phonemes, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_phonemes + 1)  # + blank

    def forward(self, x):
        # x: [B, T, D]

        out, _ = self.lstm(x)  # [B, T, 2H]
        logits = self.fc(out)  # [B, T, V+1]

        # где-то тут добавить лстм
        return logits

# Идея Гайдед Аттеншна заключается в том,
# что мы штрафуем меньше если аттеншн ложится по диагонали
# и больше если разъезжается
class GuidedAttentionLoss(nn.Module):
    def __init__(self, g=0.2):
        super().__init__()
        self.g = g # чем больше г тем меньше штраф

    def build_mask(self, target_lengths, input_lengths, max_target_len, max_input_len, device):
        """
        W[i, t] = 1 - exp(-((t/T - i/L)^2) / (2 * g^2))
        Returns a batch of masks with shape [B, L_max, T_max].
        """
        batch_size = target_lengths.size(0)

        t_positions = torch.arange(max_input_len, device=device).float().view(1, 1, max_input_len)
        i_positions = torch.arange(max_target_len, device=device).float().view(1, max_target_len, 1)

        T = input_lengths.float().view(batch_size, 1, 1).clamp(min=1.0)
        L = target_lengths.float().view(batch_size, 1, 1).clamp(min=1.0)

        norm_t = t_positions / T
        norm_i = i_positions / L

        mask = 1.0 - torch.exp(-((norm_t - norm_i) ** 2) / (2 * self.g ** 2))
        return mask

    def forward(self, attention, target_lengths, input_lengths):
        """
        attention: [B, L_max, T_max]
        """
        device = attention.device
        batch_size, max_target_len, max_input_len = attention.shape

        guided_mask = self.build_mask(
            target_lengths=target_lengths,
            input_lengths=input_lengths,
            max_target_len=max_target_len,
            max_input_len=max_input_len,
            device=device,
        )

        valid_targets = torch.arange(max_target_len, device=device).unsqueeze(0) < target_lengths.unsqueeze(1)
        valid_inputs = torch.arange(max_input_len, device=device).unsqueeze(0) < input_lengths.unsqueeze(1)
        valid_mask = valid_targets.unsqueeze(-1) & valid_inputs.unsqueeze(1)

        loss = attention * guided_mask
        loss = loss.masked_select(valid_mask)

        return loss.mean() if loss.numel() > 0 else attention.new_tensor(0.0)


class AttentionPhonemeModel(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_phonemes,
            num_layers=2,
            phoneme_emb_dim=128,
            decoder_dim=256,
            attention_dim=128,
            pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_classes = num_phonemes + 1  # keep index 0 for blank / pad / start
        self.hidden_dim = hidden_dim
        self.decoder_dim = decoder_dim

        # Прогоняет вход через лстм
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # индексы фонем (teacher forcing), они маппятся в вектора размерности phoneme_emb_dim
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=self.num_classes,
            embedding_dim=phoneme_emb_dim,
            padding_idx=pad_idx,
        )

        self.decoder_cell = nn.LSTMCell(
            input_size=phoneme_emb_dim + hidden_dim * 2,
            hidden_size=decoder_dim,
        )

        self.encoder_proj = nn.Linear(hidden_dim * 2, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.attention_score = nn.Linear(attention_dim, 1, bias=False)

        self.output_layer = nn.Linear(decoder_dim + hidden_dim * 2, self.num_classes)
        self.stop_layer = nn.Linear(decoder_dim + hidden_dim * 2, 1)

    def encode(self, x):
        encoder_outputs, _ = self.encoder(x)
        return encoder_outputs

    def _attention(self, encoder_outputs, encoder_proj, decoder_hidden, input_lengths):
        # encoder_outputs: [B, T, 2H]
        # decoder_hidden: [B, H_dec]
        decoder_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)
        energy = torch.tanh(encoder_proj + decoder_proj)
        scores = self.attention_score(energy).squeeze(-1)  # [B, T]

        max_time = encoder_outputs.size(1)
        frame_mask = torch.arange(max_time, device=encoder_outputs.device).unsqueeze(0) >= input_lengths.unsqueeze(1)
        scores = scores.masked_fill(frame_mask, -1e9)

        attention = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention

    def forward(self, x, decoder_inputs, input_lengths):
        """
        x: [B, T, D]
        decoder_inputs: [B, L]
            Teacher-forced phoneme inputs with a start token at position 0.
        input_lengths: [B]
        """
        # Декодер: LSTMCell, который на каждом шаге смотрит на:
        # предыдущий контекст из энкодера,
        # текущий входной символ(teacher forcing),
        # своe скрытое состояние.


        encoder_outputs = self.encode(x)
        encoder_proj = self.encoder_proj(encoder_outputs)

        batch_size, target_steps = decoder_inputs.shape
        device = x.device

        hidden = encoder_outputs.new_zeros(batch_size, self.decoder_dim)
        cell = encoder_outputs.new_zeros(batch_size, self.decoder_dim)
        context = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(-1))

        logits_steps = []
        stop_steps = []
        attention_steps = []

        embedded_inputs = self.phoneme_embedding(decoder_inputs)

        for step in range(target_steps):
            decoder_input = torch.cat([embedded_inputs[:, step], context], dim=-1)
            hidden, cell = self.decoder_cell(decoder_input, (hidden, cell))
            context, attention = self._attention(encoder_outputs, encoder_proj, hidden, input_lengths)

            decoder_state = torch.cat([hidden, context], dim=-1)
            step_logits = self.output_layer(decoder_state)
            stop_logit = self.stop_layer(decoder_state).squeeze(-1)
            logits_steps.append(step_logits)
            stop_steps.append(stop_logit)
            attention_steps.append(attention)

        logits = torch.stack(logits_steps, dim=1)  # [B, L, V+1]
        stop_logits = torch.stack(stop_steps, dim=1)  # [B, L]
        attention = torch.stack(attention_steps, dim=1)  # [B, L, T]
        return logits, stop_logits, attention, encoder_outputs


def monotonic_alignment_loss(attn):
    """
    attn: [B, L, T]

    Идея:
    центры внимания по соседним фонемам должны двигаться вперед по времени,
    а не назад. Если следующая фонема смотрит левее предыдущей, даем штраф.
    """
    B, L, T = attn.shape

    frame_positions = torch.arange(T, device=attn.device, dtype=attn.dtype).view(1, 1, T)
    centroid = (attn * frame_positions).sum(dim=-1)  # [B, L]
    # центр массы внимания - средняя позиция входного токена на который смотрит аттеншен
    # средняя позиция каждого последующего токена должна быть больше чем у предыдущего т е внимание идет вправо


    diff = centroid[:, :-1] - centroid[:, 1:]  # хотим <= 0
    loss = torch.relu(diff).mean()

    return loss


def train_attention_model_monotonic(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        num_epochs,
        device,
        writer,
        save_path,
        lambda_att=1.0,
):
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    stop_loss_fn = nn.BCEWithLogitsLoss()

    model.to(device)
    best_test_loss = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_total = 0.0
        train_ce = 0.0
        train_stop = 0.0
        train_att = 0.0

        for x, y, x_lens, y_lens in tqdm(train_loader, desc=f"train {epoch + 1}/{num_epochs}"):
            x = x.to(device)
            y = y.to(device)
            x_lens = x_lens.to(device)
            y_lens = y_lens.to(device)

            decoder_inputs = build_decoder_inputs(y).to(device)
            stop_targets = build_stop_targets(y_lens, y.size(1), device)
            logits, stop_logits, attention, _ = model(x, decoder_inputs, x_lens)

            ce_loss = ce_loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            stop_loss = stop_loss_fn(stop_logits, stop_targets)
            att_loss = monotonic_alignment_loss(attention)
            loss = ce_loss + stop_loss + lambda_att * att_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total += loss.item()
            train_ce += ce_loss.item()
            train_stop += stop_loss.item()
            train_att += att_loss.item()

        train_total /= len(train_loader)
        train_ce /= len(train_loader)
        train_stop /= len(train_loader)
        train_att /= len(train_loader)

        model.eval()
        test_total = 0.0
        test_ce = 0.0
        test_stop = 0.0
        test_att = 0.0

        with torch.no_grad():
            for x, y, x_lens, y_lens in tqdm(test_loader, desc=f"eval {epoch + 1}/{num_epochs}"):
                x = x.to(device)
                y = y.to(device)
                x_lens = x_lens.to(device)
                y_lens = y_lens.to(device)

                decoder_inputs = build_decoder_inputs(y).to(device)
                stop_targets = build_stop_targets(y_lens, y.size(1), device)
                logits, stop_logits, attention, _ = model(x, decoder_inputs, x_lens)

                ce_loss = ce_loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                stop_loss = stop_loss_fn(stop_logits, stop_targets)
                att_loss = monotonic_alignment_loss(attention)
                loss = ce_loss + stop_loss + lambda_att * att_loss

                test_total += loss.item()
                test_ce += ce_loss.item()
                test_stop += stop_loss.item()
                test_att += att_loss.item()

        test_total /= len(test_loader)
        test_ce /= len(test_loader)
        test_stop /= len(test_loader)
        test_att /= len(test_loader)

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("MonotonicLoss/train_total", train_total, epoch)
        writer.add_scalar("MonotonicLoss/train_ce", train_ce, epoch)
        writer.add_scalar("MonotonicLoss/train_stop", train_stop, epoch)
        writer.add_scalar("MonotonicLoss/train_att", train_att, epoch)
        writer.add_scalar("MonotonicLoss/test_total", test_total, epoch)
        writer.add_scalar("MonotonicLoss/test_ce", test_ce, epoch)
        writer.add_scalar("MonotonicLoss/test_stop", test_stop, epoch)
        writer.add_scalar("MonotonicLoss/test_att", test_att, epoch)
        writer.flush()

        print(
            f"Epoch {epoch + 1}: "
            f"train_total={train_total:.4f} | train_ce={train_ce:.4f} | train_stop={train_stop:.4f} | train_att={train_att:.4f} | "
            f"test_total={test_total:.4f} | test_ce={test_ce:.4f} | test_stop={test_stop:.4f} | test_att={test_att:.4f}"
        )

        if test_total < best_test_loss:
            best_test_loss = test_total
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "test_total": best_test_loss,
                    "lambda_att": lambda_att,
                },
                save_path,
            )
            print(f"  -> Best model saved at epoch {epoch + 1} with test_total={best_test_loss:.4f}")

    print(f"Training finished. Best test loss: {best_test_loss:.4f} at epoch {best_epoch + 1}.")


def forward_sum_loss_single(log_probs):
    """
    log_probs: [N, T]
    N = number of phonemes
    T = number of frames
    """
    N, T = log_probs.shape
    neg_inf = log_probs.new_tensor(-1e9)

    # alpha_prev хранит DP-состояние для предыдущего времени t-1
    alpha_prev = torch.full((N,), neg_inf, device=log_probs.device, dtype=log_probs.dtype)
    alpha_prev[0] = log_probs[0, 0]

    for t in range(1, T):
        alpha_cur = []

        for i in range(N):
            stay = alpha_prev[i]
            move = alpha_prev[i - 1] if i > 0 else neg_inf

            alpha_i_t = torch.logsumexp(
                torch.stack([stay, move], dim=0),
                dim=0,
            ) + log_probs[i, t]

            alpha_cur.append(alpha_i_t)

        alpha_prev = torch.stack(alpha_cur, dim=0)

    return -alpha_prev[N - 1]


def forward_sum_loss(attention, target_lengths, input_lengths, eps=1e-8):
    """
    attention: [B, L, T]
    target_lengths: [B]
    input_lengths: [B]
    """
    log_attention = torch.log(attention.clamp_min(eps))
    losses = []

    for b in range(attention.size(0)):
        L = int(target_lengths[b].item())
        T = int(input_lengths[b].item())

        if L <= 0 or T <= 0:
            continue

        # Для монотонного пути нужно хотя бы T >= L
        if T < L:
            continue

        log_probs_bt = log_attention[b, :L, :T]
        losses.append(forward_sum_loss_single(log_probs_bt))

    if len(losses) == 0:
        return attention.new_tensor(0.0)

    return torch.stack(losses).mean()

