# Отчёт: Фиксация версий зависимостей (Пункт 1.1 плана)

## Дата: 15 июля 2026

---

## 1. Что было до изменений

### Текущий `pyproject.toml` (до):
```toml
[project]
dependencies = [
    "numpy (>=2.4.3,<3.0.0)",
    "scikit-learn (>=1.8.0,<2.0.0)",
    "pandas (>=3.0.2,<4.0.0)",
    "matplotlib (>=3.10.8,<4.0.0)",
    "pandas-stubs (>=3.0.1,<3.1.0)",
    "ipykernel (>=7.3.0,<8.0.0)",
    "wordcloud (>=1.9.6,<2.0.0)"
]
```

### Проблемы:
1. **7 пакетов** в main dependencies — но проект использует **~20 сторонних пакетов**
2. Отсутствуют: `scipy`, `seaborn`, `joblib`, `tqdm`, `soundfile`, `librosa`, `lmdb`, `torch`, `torchaudio`, `transformers`, `pymorphy3`, `xgboost`, `opensmile`, `gensim`, `click`, `crowdkit`
3. Нет группировки — невозможно установить только PyTorch без opensmile
4. `pandas-stubs` и `ipykernel` были в main, хотя это dev-зависимости
5. `data_processing/requirements.txt` содержит устаревшие версии (numpy 1.21.5, pandas 1.3.5)

---

## 2. Что сделано

### 2.1 Сканирование импортов по всему проекту

Найдены все сторонние импорты в `dusha/`:

| Пакет | Где используется | Статус |
|-------|-----------------|--------|
| `numpy` | везде | ✅ был |
| `pandas` | везде | ✅ был |
| `scikit-learn` | модели, утилиты | ✅ был |
| `scipy` | audio_analise.py (DCT) | ❌ отсутствовал |
| `matplotlib` | визуализации | ✅ был |
| `seaborn` | ноутбуки | ❌ отсутствовал |
| `joblib` | model_io.py | ❌ отсутствовал |
| `tqdm` | загрузка данных | ❌ отсутствовал |
| `soundfile` | librosa dependency | ❌ отсутствовал |
| `librosa` | calculate_features.py | ❌ отсутствовал |
| `lmdb` | lmdb_utils.py | ❌ отсутствовал |
| `torch` | PyTorch модели | ❌ отсутствовал |
| `torchaudio` | аудио модели | ❌ отсутствовал |
| `transformers` | RuBERT, Wav2Vec2 | ❌ отсутствовал |
| `pymorphy3` | text_analise.py | ❌ отсутствовал |
| `wordcloud` | text_analise.py | ✅ был (в main) |
| `xgboost` | openSmile_XGBoost.py | ❌ отсутствовал |
| `opensmile` | openSmile_XGBoost.py | ❌ отсутствовал |
| `gensim` | Embeddings_LogReg.py (FastText) | ❌ отсутствовал |
| `click` | processing.py | ❌ отсутствовал |
| `crowdkit` | dawidskene.py | ❌ отсутствовал |

### 2.2 Обновлённый `pyproject.toml`

Разбит на **5 групп + dev**:

| Группа | Пакеты | Куда записано |
|--------|--------|---------------|
| **core** (main) | numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, joblib, tqdm, soundfile, librosa, lmdb | `[project].dependencies` |
| **ml** | torch, torchaudio, transformers | `[project.optional-dependencies].ml` |
| **analysis** | pymorphy3, pymorphy3-dicts-ru, wordcloud | `[project.optional-dependencies].analysis` |
| **audio-extra** | opensmile, xgboost | `[project.optional-dependencies].audio-extra` |
| **text-extra** | gensim | `[project.optional-dependencies].text-extra` |
| **processing** | click, crowd-kit | `[project.optional-dependencies].processing` |
| **dev** | setuptools, ipykernel, pandas-stubs, pytest, ruff | `[dependency-groups].dev` |

### 2.3 Исправления

- `crowdkit` → `crowd-kit` (правильное имя пакета на PyPI)
- `wordcloud` перенесён из main в группу `analysis` (нужен только для EDA)
- `pandas-stubs` и `ipykernel` перенесены в dev
- Добавлены `pytest` и `ruff` в dev

---

## 3. Как пользоваться

### Установка базовых зависимостей (для запуска большинства экспериментов):
```bash
poetry install
```
Установит: numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, joblib, tqdm, soundfile, librosa, lmdb + dev (pytest, ruff).

### Установка с PyTorch + Transformers:
```bash
poetry install --extras ml
```
Добавит: torch, torchaudio, transformers.

### Установка с анализом данных (для ноутбуков):
```bash
poetry install --extras analysis
```
Добавит: pymorphy3, pymorphy3-dicts-ru, wordcloud.

### Установка всего сразу:
```bash
poetry install --extras "ml analysis audio-extra text-extra processing"
```

### Установка конкретных групп:
```bash
# Только для работы с данными (processing pipeline)
poetry install --extras processing

# Только для аудио моделей (openSMILE + XGBoost)
poetry install --extras audio-extra

# Только для текстовых моделей (FastText)
poetry install --extras text-extra
```

---

## 4. Файлы, которые были изменены

| Файл | Изменение |
|------|-----------|
| `pyproject.toml` | Полностью переписан: добавлены группы зависимостей, исправлены имена пакетов |
| `poetry.lock` | Перегенерирован (`poetry lock`) — теперь содержит все зависимости с точными версиями |

---

## 5. Проверка

```bash
# Валидация конфига
poetry check  # → All set!

# Импорт всех core пакетов
python -c "import numpy, pandas, sklearn, scipy, matplotlib, seaborn, joblib, tqdm, soundfile, librosa, lmdb"  # → OK

# Импорт ML пакетов
python -c "import torch, torchaudio, transformers"  # → OK

# Импорт analysis пакетов
python -c "import pymorphy3, wordcloud"  # → OK
```

---

## 6. Статус: ✅ ВЫПОЛНЕНО

Все зависимости зафиксированы, сгруппированы и проверены.
