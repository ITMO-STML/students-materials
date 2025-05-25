from typing import List
from razdel import tokenize, substring
from string import punctuation
from copy import deepcopy


class DiarizationCorrector:
    """
    Corrects diarization mistakes by merging short utterances 
    with the neighbouring utterances
    """
    def __init__(
            self
    ):
        pass

    def remove_punct_from_tokens(
            self,
            tokens : List[substring.Substring],
    ) -> List[str]:
        """
        Removes punctuation from tokens;
        Returns list of tokens as strings
        """
        tokens_wo_punct = []
        for substr in tokens:
            if substr and substr.text not in punctuation:
                tokens_wo_punct.append(substr.text)
        return tokens_wo_punct
    
    def tokenize_wo_punct(
            self,
            utterance: dict
    ) -> List[str]:
        """
        Tokenizes utterance
        """
        utterance_tokens = list(tokenize(utterance))
        return self.remove_punct_from_tokens(utterance_tokens)

    def correct_diar_mistakes(
        self,
        chunk : dict
    ) -> List[dict]:
        """
        Corrects diarization mistakes in the current chunk
        """
        chunk = deepcopy(chunk["transcript"])
        new_diar = []
        for i in range(len(chunk)):
            chunk_tokens = self.tokenize_wo_punct(chunk[i]["text"])
            if new_diar:
                if len(chunk_tokens) <= 2 or new_diar[-1]["speaker"] == chunk[i]["speaker"]:
                    new_diar[-1]["text"] = f"{new_diar[-1]["text"]} {chunk[i]["text"]}"
                    new_diar[-1]["endTime"] = chunk[i]["endTime"]
                    # Целевой диктор – тот, у кого длиннее реплика из текущей пары реплик
                    if len(new_diar[-1]["text"]) <= len(chunk[i]["text"]):
                        new_diar[-1]["speaker"] = chunk[i]["speaker"]
                else:
                    new_diar.append(chunk[i])
            elif len(chunk_tokens) <= 2:
                chunk[i+1]["text"] = f"{chunk[i]["text"]} {chunk[i+1]["text"]}"
                chunk[i+1]["startTime"] = chunk[i]["startTime"]
            else:
                new_diar.append(chunk[i])
        return new_diar
        