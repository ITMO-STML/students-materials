# Speech Intelligibility Prediction

This tool predicts speech intelligibility (STOI score) and SNR from microphone input using a deep learning model.

## Requirements

- Python 3.8+
- Libraries from `requirements.txt`:

  ```
  numpy
  torch
  sounddevice
  librosa
  soundfile
  pystoi
  scipy
  matplotlib
  noisereduce
  ```

## Installation

1. Clone the repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python3 microphone_test_new_model.py
```

Follow the on-screen instructions:
- Press Enter to start recording (10 seconds)
- The system will analyze speech segments and display:
  - STOI score (speech intelligibility)
  - SNR (signal-to-noise ratio)
  - Quality assessment

Press 'q' to quit.

## Output

The program will:
1. Show real-time analysis of each speech segment
2. Display average STOI and SNR for the recording

Note: For best results, use a quiet environment and speak clearly into the microphone.

