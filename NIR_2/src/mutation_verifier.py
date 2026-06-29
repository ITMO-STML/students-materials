"""
Mutation Verification (MV).

Задача CN-пайплайна: построить d⁻ из d⁺ путём замены конкретного факта.
Раньше валидность d⁻ проверялась судьями (CE + LLM), которые отвечают
на вопрос «является ли d⁻ ответом на q?». Это требует от судьи внешнего
ground truth (priors модели по теме). На knowledge-tail (даты/числа) priors
у Qwen2.5-7B нестабильны → CoT-judge даёт 41.9% accuracy на CF, причём
**ошибки скоррелированы с ошибками CE** по типу мутации (на датах оба слепы).

MV меняет фрейм. Вместо вопроса «is d⁻ wrong about q?» мы спрашиваем
«was the mutation carried out faithfully?». Это проверяется детерминистически
по построению — у нас есть fact_text (что просили заменить) и d⁺ / d⁻.

ПРАВИЛА ВАЛИДНОЙ МУТАЦИИ:
  1. fact_text был в d⁺ (точная подстрока).
  2. Diff между d⁺ и d⁻ пересекается с позицией fact_text в d⁺
     (мутация выполнена ИМЕННО на этом факте, не где-то ещё).
  3. На месте fact_text в d⁻ стоит непустая замена, отличная от fact_text.
  4. Изменение локализовано: доля изменённых символов ≤ max_locality_ratio
     (LLM не сорвалась в overhaul).

ЧТО МЫ НАМЕРЕННО НЕ ПРОВЕРЯЕМ:
  - "fact_text больше не встречается нигде в d⁻". Substring-match даёт
    false positives на морфологии ("струн" ⊂ "струны"). Полагаемся
    на правило 2: важно что произошло В ПОЗИЦИИ fact_text, а не global.
  - "diff состоит из одного блока". Умная LLM делает связные правки
    (поменяла город → поменяла штат). Это не bug, это feature.
    Cap на blocks — soft, основной gate — locality_ratio.

Калибровка по Phase 0.7 (31 sanity CF):
  - max_locality_ratio=0.30, max_diff_blocks=12 → ожидаемый valid_rate ~93-97%.
  - Изначальные параметры (0.30, 4) давали 81% из-за слишком строгого
    blocks-cap на connected edits.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Optional


@dataclass
class MutationVerdict:
    """Результат верификации одной мутации d⁺ → d⁻."""
    valid: bool
    reason: Optional[str]            # короткий код причины, если invalid
    replacement: Optional[str]        # что стало на месте fact_text в d⁻
    fact_in_dplus: bool               # fact_text найден в d⁺
    fact_in_dminus_global: bool       # fact_text встречается где-то в d⁻ (диагностика, не fail)
    diff_at_fact_position: bool       # diff пересекается с fact_text
    locality_ratio: float             # доля изменённых символов, [0, 1]
    n_diff_blocks: int                # сколько блоков изменения
    fact_position_in_dplus: int       # стартовая позиция fact_text в d⁺ (-1 если нет)

    def to_dict(self) -> dict:
        return asdict(self)


class MutationVerifier:
    """
    Детерминистическая верификация выполнения CN-мутации.

    Концепция: мутация считается faithful, если diff между d⁺ и d⁻ пересекается
    с позицией fact_text в d⁺ И заменяет его на что-то отличное. Глобальное
    присутствие fact_text где-то ещё в d⁻ — это не fail (часто срабатывает
    substring-match: "струн" ⊂ "струны"); мы смотрим только на позицию замены.

    Параметры:
      max_locality_ratio: верхний предел на долю изменённых символов.
        Это основной gate от overhaul. 0.30 значит: если изменилось больше
        30% объёма документа, LLM ушла в переписывание → invalid.
      max_diff_blocks: верхний предел на число блоков diff. Soft cap для
        пограничных overhaul cases. По умолчанию 12 — достаточно для умных
        connected edits (поменяли город → поменяли штат → согласовали падежи).
    """

    REASON_FACT_NOT_IN_DPLUS    = "fact_text_not_in_dplus"
    REASON_NO_DIFF              = "documents_identical"
    REASON_DIFF_NOT_AT_FACT     = "diff_not_overlapping_fact"
    REASON_EMPTY_REPLACEMENT    = "empty_replacement"
    REASON_REPLACEMENT_EQUALS   = "replacement_equals_fact"
    REASON_NOT_LOCAL            = "mutation_not_local"
    REASON_TOO_MANY_BLOCKS      = "too_many_diff_blocks"

    def __init__(
        self,
        max_locality_ratio: float = 0.30,
        max_diff_blocks: int = 12,
    ):
        self.max_locality_ratio = max_locality_ratio
        self.max_diff_blocks = max_diff_blocks

    def verify(
        self, d_plus: str, d_minus: str, fact_text: str
    ) -> MutationVerdict:
        # 1. fact_text был в d⁺ (точное вхождение)
        pos = d_plus.find(fact_text)
        if pos < 0:
            return MutationVerdict(
                valid=False, reason=self.REASON_FACT_NOT_IN_DPLUS,
                replacement=None,
                fact_in_dplus=False, fact_in_dminus_global=(fact_text in d_minus),
                diff_at_fact_position=False,
                locality_ratio=0.0, n_diff_blocks=0,
                fact_position_in_dplus=-1,
            )
        fact_end = pos + len(fact_text)
        fact_in_dminus = fact_text in d_minus  # только диагностика, не gate

        # 2. Diff между d⁺ и d⁻
        sm = SequenceMatcher(None, d_plus, d_minus, autojunk=False)
        opcodes = sm.get_opcodes()
        non_equal = [op for op in opcodes if op[0] != "equal"]

        if not non_equal:
            return MutationVerdict(
                valid=False, reason=self.REASON_NO_DIFF,
                replacement=None,
                fact_in_dplus=True, fact_in_dminus_global=fact_in_dminus,
                diff_at_fact_position=False,
                locality_ratio=0.0, n_diff_blocks=0,
                fact_position_in_dplus=pos,
            )

        # 3. Найти opcodes, перекрывающиеся с диапазоном fact_text в d⁺.
        # Для opcode (tag, i1, i2, j1, j2): на стороне d⁺ это [i1, i2).
        overlapping = [
            op for op in non_equal
            if not (op[2] <= pos or op[1] >= fact_end)
        ]

        if not overlapping:
            return MutationVerdict(
                valid=False, reason=self.REASON_DIFF_NOT_AT_FACT,
                replacement=None,
                fact_in_dplus=True, fact_in_dminus_global=fact_in_dminus,
                diff_at_fact_position=False,
                locality_ratio=self._locality(non_equal, d_plus, d_minus),
                n_diff_blocks=len(non_equal),
                fact_position_in_dplus=pos,
            )

        # 4. Сбор замены (что встало на месте fact_text в d⁻)
        # Берём union j-диапазонов перекрывающихся opcodes:
        j_lo = min(op[3] for op in overlapping)
        j_hi = max(op[4] for op in overlapping)
        replacement = d_minus[j_lo:j_hi].strip()

        # Защита от случая, когда overlapping opcode — pure 'delete' (j_lo==j_hi);
        # тогда расширим j-окно соседним 'insert', если он рядом.
        if j_lo == j_hi:
            for tag, i1, i2, ji1, ji2 in non_equal:
                if tag == "insert" and (i1 == pos or i1 == fact_end):
                    replacement = d_minus[ji1:ji2].strip()
                    j_lo, j_hi = ji1, ji2
                    break

        if not replacement:
            return MutationVerdict(
                valid=False, reason=self.REASON_EMPTY_REPLACEMENT,
                replacement="",
                fact_in_dplus=True, fact_in_dminus_global=fact_in_dminus,
                diff_at_fact_position=True,
                locality_ratio=self._locality(non_equal, d_plus, d_minus),
                n_diff_blocks=len(non_equal),
                fact_position_in_dplus=pos,
            )

        if replacement == fact_text:
            return MutationVerdict(
                valid=False, reason=self.REASON_REPLACEMENT_EQUALS,
                replacement=replacement,
                fact_in_dplus=True, fact_in_dminus_global=fact_in_dminus,
                diff_at_fact_position=True,
                locality_ratio=self._locality(non_equal, d_plus, d_minus),
                n_diff_blocks=len(non_equal),
                fact_position_in_dplus=pos,
            )

        # 5. Локализованность изменения
        locality = self._locality(non_equal, d_plus, d_minus)
        n_blocks = len(non_equal)

        if locality > self.max_locality_ratio:
            return MutationVerdict(
                valid=False, reason=self.REASON_NOT_LOCAL,
                replacement=replacement,
                fact_in_dplus=True, fact_in_dminus_global=fact_in_dminus,
                diff_at_fact_position=True,
                locality_ratio=locality, n_diff_blocks=n_blocks,
                fact_position_in_dplus=pos,
            )
        if n_blocks > self.max_diff_blocks:
            return MutationVerdict(
                valid=False, reason=self.REASON_TOO_MANY_BLOCKS,
                replacement=replacement,
                fact_in_dplus=True, fact_in_dminus_global=fact_in_dminus,
                diff_at_fact_position=True,
                locality_ratio=locality, n_diff_blocks=n_blocks,
                fact_position_in_dplus=pos,
            )

        return MutationVerdict(
            valid=True, reason=None,
            replacement=replacement,
            fact_in_dplus=True, fact_in_dminus_global=fact_in_dminus,
            diff_at_fact_position=True,
            locality_ratio=locality, n_diff_blocks=n_blocks,
            fact_position_in_dplus=pos,
        )

    @staticmethod
    def _locality(non_equal_ops, d_plus: str, d_minus: str) -> float:
        """
        Доля символов, изменившихся в d⁺ ∪ d⁻.
        0 — ничего не изменилось, 1 — всё изменилось.
        Считаем сумму max(len_left, len_right) каждого не-equal блока,
        делим на максимум длин d⁺ и d⁻ (а не на сумму, чтобы 1.0 был
        достижимым максимумом, когда замена кардинальная).
        """
        if not d_plus or not d_minus:
            return 1.0
        changed = 0
        for tag, i1, i2, j1, j2 in non_equal_ops:
            changed += max(i2 - i1, j2 - j1)
        return changed / max(len(d_plus), len(d_minus))
