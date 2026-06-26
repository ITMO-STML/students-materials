class MetricsCalculator:
    @staticmethod
    def normalize(text):
        return (
            str(text)
            .replace("_", " ")
            .replace('"', "")
            .replace("é", "e")
            .replace("–", "-")
            .strip()
            .lower()
        )

    @classmethod
    def normalize_triple(cls, triple):
        return (
            cls.normalize(triple["sub"]),
            str(triple["rel"]).strip(),
            cls.normalize(triple["obj"])
        )

    @classmethod
    def compute_strict_metrics(cls, predicted, gold):
        predicted_set = set(
            cls.normalize_triple(t)
            for t in predicted
            if "sub" in t and "rel" in t and "obj" in t
        )

        gold_set = set(
            cls.normalize_triple(t)
            for t in gold
            if "sub" in t and "rel" in t and "obj" in t
        )

        correct = predicted_set & gold_set

        precision = len(correct) / len(predicted_set) if predicted_set else 0
        recall = len(correct) / len(gold_set) if gold_set else 0

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "correct": len(correct),
            "predicted": len(predicted_set),
            "gold": len(gold_set),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    @classmethod
    def token_set(cls, text):
        normalized = cls.normalize(text)

        for ch in [
            ",", ".", ":", ";", "(", ")", "[", "]",
            "{", "}", "-", "/", "\\"
        ]:
            normalized = normalized.replace(ch, " ")

        return set(
            token
            for token in normalized.split()
            if token
        )

    @classmethod
    def object_overlap_score(cls, pred_obj, gold_obj):
        pred_tokens = cls.token_set(pred_obj)
        gold_tokens = cls.token_set(gold_obj)

        if not pred_tokens or not gold_tokens:
            return 0.0

        overlap = pred_tokens & gold_tokens

        return len(overlap) / len(gold_tokens)

    @classmethod
    def subject_match(cls, pred_sub, gold_sub):
        return cls.normalize(pred_sub) == cls.normalize(gold_sub)

    @classmethod
    def compute_fact_metrics(cls, predicted, gold):
        captured_gold = 0
        matched_predictions = set()

        for gold_triple in gold:
            gold_sub = gold_triple.get("sub", "")
            gold_obj = gold_triple.get("obj", "")

            found = False

            for i, pred_triple in enumerate(predicted):
                if i in matched_predictions:
                    continue

                pred_sub = pred_triple.get("sub", "")
                pred_obj = pred_triple.get("obj", "")

                if not cls.subject_match(pred_sub, gold_sub):
                    continue

                overlap = cls.object_overlap_score(pred_obj, gold_obj)

                if overlap >= 0.8:
                    found = True
                    matched_predictions.add(i)
                    break

            if found:
                captured_gold += 1

        fact_recall = captured_gold / len(gold) if gold else 0.0

        fact_precision = (
            len(matched_predictions) / len(predicted)
            if predicted else 0.0
        )

        if fact_precision + fact_recall == 0:
            fact_f1 = 0.0
        else:
            fact_f1 = (
                2 * fact_precision * fact_recall /
                (fact_precision + fact_recall)
            )

        return {
            "captured_gold": captured_gold,
            "matched_predictions": len(matched_predictions),
            "predicted": len(predicted),
            "gold": len(gold),
            "fact_precision": fact_precision,
            "fact_recall": fact_recall,
            "fact_f1": fact_f1
        }