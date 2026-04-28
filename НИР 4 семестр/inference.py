import numpy as np

import math

class Inferencer():
    def __init__(self, model):
        self.model = model
        self.shift_delimeter = 2
    
    def inference(self, sound_data, sample_rate):
        data_len = len(sound_data)
        sample_len = sample_rate
        sample_shift = sample_rate // self.shift_delimeter
        max_shift = (data_len - sample_len + sample_shift - 1) // sample_shift * sample_shift
        sound_samples = [sound_data[i:min(i + sample_len, data_len)].tolist() for i in range(0, max_shift + 1, sample_shift)]
        sound_samples[-1] = sound_samples[-1] + [0] * (sample_len - len(sound_samples[-1]))
        sound_samples = np.array(sound_samples, dtype = "float32")

        scores = []
        for sound_sample in sound_samples:
            score = self.model.run(None, {"waves": [sound_sample]})[0][0]

            score = 1 / (1 + math.exp(-(score[1] - score[0])))
            scores.append(score)

        return scores