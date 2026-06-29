# Настройка окружения (Windows)

## Что установлено

| Компонент | Путь / Версия |
|-----------|---------------|
| Python 3.13 | `C:\Users\Александр\AppData\Local\Programs\Python\Python313\` |
| Ollama 0.30.10 | стандартная установка |
| Модели Ollama | `W:\ollama_home\models\` (junction от `~\.ollama`) |
| GGUF бэкап | `W:\models\qwen2.5-7b-q4.gguf` |
| HuggingFace кэш | `W:\huggingface_cache\` |
| Проект NIR | `W:\Jupyter\NIR_2\` |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM, CUDA 12.7 |

## После каждой перезагрузки

### Терминал 1 (сервер — не закрывать):

```
W:\Jupyter\NIR_2\scripts\start_env.bat
```

Или вручную:
```powershell
Get-Process *ollama* | Stop-Process -Force
Start-Sleep 3
$env:OLLAMA_MODELS = "W:\ollama_home\models"
ollama serve
```

### Терминал 2 (работа):

```
W:\Jupyter\NIR_2\scripts\run_sanity.bat
```

Или вручную:
```powershell
$env:HF_HOME = "W:\huggingface_cache"
python W:\Jupyter\NIR_2\scripts\00_sanity_qwen.py
```

## Почему так сложно

Имя пользователя `Александр` содержит кириллицу. llama-server (C++)
не умеет открывать файлы по пути с Unicode. Ollama как Windows-сервис
берёт путь из `~\.ollama` = `C:\Users\Александр\.ollama` — и падает.

Решение: запускаем `ollama serve` вручную из терминала с
`$env:OLLAMA_MODELS = "W:\ollama_home\models"` — тогда путь ASCII.

## Если что-то не работает

1. **Ollama не стартует (порт занят):**
   ```powershell
   Get-Process *ollama* | Stop-Process -Force
   Start-Sleep 3
   # повторить start_env.bat
   ```

2. **"model not found":**
   ```powershell
   $env:OLLAMA_MODELS = "W:\ollama_home\models"
   ollama list            # проверить что модель видна
   ollama pull qwen2.5:7b # если нет — скачать
   ```

3. **Ошибки с HuggingFace кэшем:**
   ```powershell
   $env:HF_HOME = "W:\huggingface_cache"
   ```

4. **Хочу GPU для llama-cpp-python (вместо Ollama):**
   Нужен CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
   После установки:
   ```powershell
   $env:CMAKE_ARGS = "-DGGML_CUDA=on"
   pip install llama-cpp-python --force-reinstall --no-binary llama-cpp-python
   ```
