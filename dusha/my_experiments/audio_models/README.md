# Baseline: Логистическая регрессия

## Описание
Скрипт для обучения и оценки логистической регрессии для классификации эмоций.

## Использование

### Автоматический режим (по умолчанию)
Загружает модель если существует, иначе обучает новую:
```bash
poetry run python dusha/my_experiments/audio_models/baseline/logistic_regression.py
```

### Обучить новую модель
Всегда обучает модель заново (перезаписывает существующую):
```bash
poetry run python dusha/my_experiments/audio_models/baseline/logistic_regression.py --mode train
```

### Загрузить существующую модель
Только загружает и оценивает существующую модель:
```bash
poetry run python dusha/my_experiments/audio_models/baseline/logistic_regression.py --mode load
```

### Обучить без сохранения
Обучает модель, но не сохраняет результат:
```bash
poetry run python dusha/my_experiments/audio_models/baseline/logistic_regression.py --mode train --no-save
```

## Сохранение модели

Модели сохраняются в `dusha/my_experiments/audio_models/baseline/models_params/` с указанием датасета:
- `logistic_regression_combine_balanced_train_small_model.pkl` - модель
- `logistic_regression_combine_balanced_train_small_scaler.pkl` - нормализатор
- `logistic_regression_combine_balanced_train_small_model_YYYYMMDD_HHMMSS.pkl` - бэкап с временной меткой

Имя файла формируется как: `{имя_скрипта}_{имя_датасета}_{тип}.pkl`

## Вывод

Скрипт выводит:
- ✅ Количество загруженных примеров
- ✅ Распределение классов
- ✅ Параметры обученной модели
- ✅ Метрики качества (precision, recall, f1-score)
- ✅ Confusion matrix
- ✅ Путь к сохранённым файлам
