import math
from typing import Dict, List


def get_phrase(list_of_phonemes): # removes repeated
    prev_let = None
    phrase = ''
    phrase_list = []
    for letter in list_of_phonemes:
        if letter != prev_let:
            phrase += letter
            prev_let = letter
            phrase_list.append(letter)
    return phrase, phrase_list


def viterbi_decode(
    probability_dist: Dict[str, List[Dict[str, float]]],
    alpha: float = 0.4,
    lambda_penalty: float = 1.0,
    eps: float = 1e-8,
):
    """
    Viterbi decoding for frame-level phoneme alignment using
    start / center / end probability distributions.

    Args:
        probability_dist: dict with keys ["start", "center", "end"],
            each is a list of length T with dicts {phoneme: probability}
        alpha: weight for start/end context
        lambda_penalty: transition penalty for phoneme change
        eps: small value to avoid log(0)

    Returns:
        path: List[str] of length T (decoded phoneme per frame)
        boundaries: List[int] frame indices where phoneme changes
    """

    # ----------------------------
    # 1. Collect phoneme inventory
    # ----------------------------
    phonemes = set()
    for part in ["start", "center", "end"]:
        for frame in probability_dist[part]:
            phonemes |= frame.keys()
    phonemes = sorted(phonemes)

    T = len(probability_dist["center"])
    assert all(len(probability_dist[p]) == T for p in ["start", "center", "end"])

    # ----------------------------
    # 2. Helper: log-probability
    # ----------------------------
    def logp(frame_dict, phoneme):
        return math.log(frame_dict.get(phoneme, eps))

    # ----------------------------
    # 3. Allocate DP tables
    # ----------------------------
    delta = [dict() for _ in range(T)]  # best score
    psi = [dict() for _ in range(T)]    # backpointers

    # ----------------------------
    # 4. Initialization
    # ----------------------------
    for ph in phonemes:
        delta[0][ph] = (
            logp(probability_dist["center"][0], ph)
            + alpha * logp(probability_dist["start"][0], ph)
            + alpha * logp(probability_dist["end"][0], ph)
        )
        psi[0][ph] = None

    # ----------------------------
    # 5. Viterbi recursion
    # ----------------------------
    for t in range(1, T):
        for ph in phonemes:
            emission = (
                logp(probability_dist["center"][t], ph)
                + alpha * logp(probability_dist["start"][t], ph)
                + alpha * logp(probability_dist["end"][t], ph)
            )

            best_score = -float("inf")
            best_prev = None

            for prev_ph in phonemes:
                transition = 0.0 if prev_ph == ph else -lambda_penalty
                score = delta[t - 1][prev_ph] + transition

                if score > best_score:
                    best_score = score
                    best_prev = prev_ph

            delta[t][ph] = best_score + emission
            psi[t][ph] = best_prev

    # ----------------------------
    # 6. Backtracking
    # ----------------------------
    last_ph = max(delta[T - 1], key=lambda p: delta[T - 1][p])
    path = [last_ph]

    for t in reversed(range(1, T)):
        path.append(psi[t][path[-1]])

    path.reverse()

    # ----------------------------
    # 7. Extract boundaries
    # ----------------------------
    boundaries = [t for t in range(1, T) if path[t] != path[t - 1]]

    phonemes_no_rep = [path[0]]
    for i in range(1, len(path)):
        if path[i] != path[i - 1]:
            phonemes_no_rep.append(path[i])

    return path, phonemes_no_rep, boundaries


def viterbi_lookahead(
    probability_dist,
    alpha=0.4,
    lambda_penalty=3.0,
    lookahead=2,
    gamma=0.7,
    log_floor=-20.0,
):
    """
    Viterbi с учётом будущих кадров (lookahead).

    probability_dist: dict с keys ['start','center','end'], каждый список длины T
    alpha: вес start/end
    lambda_penalty: штраф за смену фонемы
    lookahead: количество будущих кадров, которые учитываем
    gamma: коэффициент затухания влияния будущих кадров
    log_floor: минимальное значение log
    """

    # --- фонемы ---
    phonemes = set()
    for part in ["start","center","end"]:
        for frame in probability_dist[part]:
            phonemes.update(frame.keys())
    phonemes = sorted(phonemes)

    T = len(probability_dist["center"])
    delta = [dict() for _ in range(T)]
    psi = [dict() for _ in range(T)]

    def logp(frame, ph):
        return math.log(frame[ph]) if ph in frame else log_floor

    # --- эмиссия с lookahead ---
    emission = [dict() for _ in range(T)]
    for t in range(T):
        for ph in phonemes:
            score = (
                logp(probability_dist["center"][t], ph)
                + alpha*logp(probability_dist["start"][t], ph)
                + alpha*logp(probability_dist["end"][t], ph)
            )
            # lookahead
            for i in range(1, lookahead+1):
                if t + i < T:
                    score += (gamma**i) * (
                        logp(probability_dist["center"][t+i], ph)
                        + alpha*logp(probability_dist["start"][t+i], ph)
                        + alpha*logp(probability_dist["end"][t+i], ph)
                    )
            emission[t][ph] = score

    # --- инициализация ---
    for ph in phonemes:
        delta[0][ph] = emission[0][ph]
        psi[0][ph] = None

    # --- рекурсия ---
    for t in range(1, T):
        for ph in phonemes:
            best_score, best_prev = -1e18, None
            for prev_ph in phonemes:
                transition = 0.0 if prev_ph == ph else -lambda_penalty
                score = delta[t-1][prev_ph] + transition
                if score > best_score:
                    best_score = score
                    best_prev = prev_ph
            delta[t][ph] = best_score + emission[t][ph]
            psi[t][ph] = best_prev

    # --- backtracking ---
    last_ph = max(delta[-1], key=delta[-1].get)
    path = [last_ph]
    for t in reversed(range(1, T)):
        path.append(psi[t][path[-1]])
    path.reverse()

    # --- границы ---
    boundaries = [t for t in range(1, T) if path[t] != path[t-1]]

    phonemes_no_rep = [path[0]]
    for i in range(1, len(path)):
        if path[i] != path[i - 1]:
            phonemes_no_rep.append(path[i])

    return path, phonemes_no_rep, boundaries


def write_to_seg(name, boundaries, res, level=1024 ):

    pred_labelling = []

    with open(name, 'w') as f:
        f.write('[PARAMETERS]')
        f.write('\n')
        f.write('SAMPLING_FREQ=22050')
        f.write('\n')
        f.write('BYTE_PER_SAMPLE=2')
        f.write('\n')
        f.write('CODE=0')
        f.write('\n')
        f.write('N_CHANNEL=1')
        f.write('\n')
        f.write(f'N_LABEL={len(boundaries)}')
        f.write('\n')
        f.write('[LABELS]')
        f.write('\n')
        for ph, b in zip(res, boundaries):
            time = float(b) * 0.02 * 22050 * 2
            f.write(f'{int(time)}, {level}, {ph}\n')
            pred_labelling.append((float(b) * 0.02, ph))

    print(f'{name} is ready')
    return pred_labelling



