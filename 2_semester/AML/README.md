# Hear-deasis classification

## Data
Download a `.csv` file with data. It should contain columns `Sex` (0 - male, 1 - female), `Age`, `Cholesterol`, `Max HR` and "Heart Disease" (`Presence` and `Absense`). [Original data](https://www.kaggle.com/competitions/playground-series-s6e2/data?select=train.csv)

## Scripts
### Env
You can use `uv` (prefered) or `pip` as a dependency manager to init venv:

```shell
uv sync --locked

# or

pip install .
```
Alternalively you can use `Docker`
```shell
docker build .
```

### Training
To get model's state, you can run `train.py` script.
```shell
uv run train.py \
    --df_path /path/to/csv
    --state_save_path /path/to/json
    --seed 42
# or
python train.py \
    --df_path /path/to/csv
    --state_save_path /path/to/json
    --seed 42
# or Docker
docker run -it /bin/bash -c "uv run gui.py"
```
### GUI
To launch gradio app `gui.py` with
```shell
uv run gui.py \
    --state_path /path/to/json
# or
python train.py \
    --state_save_path /path/to/json
# or Docker
docker run -p 7860:7860 -it $(docker build -q .) /bin/bash -c "uv run gui.py"
```