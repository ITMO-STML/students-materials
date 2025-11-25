import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
import IPython

import sys
sys.path.insert(0, "melgan")
sys.path.insert(0, "diffwave/src")
sys.path.insert(0, "wavegrad/src")
sys.path.insert(0, "WaveRNN")

from speechbrain.inference.TTS import MSTacotron2
from speechbrain.inference.vocoders import HIFIGAN, DiffWaveVocoder
from WaveRNN import inference

DATA_ROOT = "/mnt/hot_store/spoofbase/data/LibriTTS"
SAVE_ROOT = "/mnt/hot_store/spoofbase/purgatory/LibriTTS_vocoding"

libri_tts_df = pd.read_csv("/mnt/cold_store/antispoofing/data/orig/Libri_TTS/text_metadata.tsv", sep='\t')
libri_tts_test_df = libri_tts_df[libri_tts_df["wav_path"].apply(lambda x: "test" in x)]

samples = libri_tts_test_df.sample(3000, random_state=100)

def clone_voice(tacotron, vocoder, vocoder_sr, target_sr, input_text, reference_speech):
    mel_outputs, mel_lengths, alignments = tacotron.clone_voice(input_text, reference_speech)
    waveforms = vocoder.decode_batch(mel_outputs, hop_len=256, fast_sampling=True, fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])
    return torchaudio.functional.resample(waveforms, vocoder_sr, target_sr), target_sr

def clone_voices(tacotron, vocoder, vocoder_sr, target_sr, samples, tts_vocoder_root):
    for index, raw in tqdm(samples.iterrows()):
        save_path = os.path.join(SAVE_ROOT, tts_vocoder_root, f"{raw['wav_id']}.wav")
        if os.path.exists(save_path):
            continue
        
        try:
            wav, sr = clone_voice(tacotron, vocoder, vocoder_sr, target_sr, raw["text_orig"], os.path.join(DATA_ROOT, raw["wav_path"]))
            torchaudio.save(save_path, wav.squeeze(1), sample_rate=sr)
        except RuntimeError as e:
            print("Something went wrong while processing:", raw["wav_id"])
            print(str(e))

TTS_VOCODER_ROOT = "ms_tacotron2/hifi_gan"
ms_tacotron2 = MSTacotron2.from_hparams(source="speechbrain/tts-mstacotron2-libritts", savedir="pretrained_models/tts-mstacotron2-libritts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="pretrained_models/tts-hifigan-libritts-22050Hz")
ms_tacotron2.eval()
hifi_gan.eval()
clone_voices(ms_tacotron2, hifi_gan, 22050, 16000, samples, TTS_VOCODER_ROOT)


TTS_VOCODER_ROOT = "ms_tacotron2/diffwave__ljspeech"
ms_tacotron2 = MSTacotron2.from_hparams(source="speechbrain/tts-mstacotron2-libritts", savedir="pretrained_models/tts-mstacotron2-libritts")
diffwave = DiffWaveVocoder.from_hparams(source="speechbrain/tts-diffwave-ljspeech", savedir="pretrained_models/tts-diffwave-ljspeech")
ms_tacotron2.eval()
diffwave.eval()
clone_voices(ms_tacotron2, diffwave, 22050, 16000, samples, TTS_VOCODER_ROOT)


TTS_VOCODER_ROOT = "tacotron_WaveRNN/WaveRNN"
for index, row in tqdm(samples.iterrows()):
    save_path = os.path.join(SAVE_ROOT, TTS_VOCODER_ROOT, f"{row['wav_id']}.wav")
    if os.path.exists(save_path):
        continue
    
    try:
        wav = inference.run_default_tts(row["text_orig"])
        torchaudio.save(save_path, torch.tensor(wav, dtype=float).unsqueeze(0), sample_rate=22050)
        IPython.display.clear_output()
    except RuntimeError as e:
        with open("wavernn_errors.txt", 'a') as f:
            f.write("Something went wrong while processing:", raw["wav_id"], '\n')
            f.write(str(e), '\n')
            f.write('\n')