# Математический практикум
## Поиск потерянной физики

# Universal Differential Equations для ядерной физики

Эта реализация позволяет моделировать взаимодействие ядерных частиц с помощью Universal Differential Equations (UDE), обучая нейронную сеть воспроизводить потенциал взаимодействия.

## Структура проекта

- `configs/` - конфигурационные файлы
- `core/` - ядро симуляции
- `potential/` - реализация аналитического потенциала
- `ude/` - модели UDE и модули для обучения
- `utils/` - вспомогательные функции
- `scripts/` - скрипты для генерации данных, обучения и анализа

## Требования

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- scikit-learn
- tqdm

Установка зависимостей:
```bash
pip install torch numpy matplotlib scikit-learn tqdm
```

Для использования KAN (Kolmogorov-Arnold Networks):
```bash
pip install pykan
```

## Запуск

### 1. Генерация данных

Генерирует данные из симуляций, которые будут использоваться для обучения UDE модели:

```bash
python main.py --mode generate_data --config configs/base_config.yaml
```

### 2. Обучение UDE

Обучает UDE модель на основе сгенерированных данных:

```bash
python main.py --mode train_ude --config configs/base_config.yaml
```

### 3. Анализ результатов

Анализирует результаты обученной модели:

```bash
python main.py --mode analyze_results --config configs/base_config.yaml
```

### 4. Запуск симуляции с обученной моделью

Запускает симуляцию с использованием обученной модели:

```bash
python main.py --mode run_simulation --config configs/base_config.yaml
```

## Настройка параметров

Все параметры можно настроить в файле конфигурации `configs/base_config.yaml` или передать аргументами командной строки:

```bash
python main.py --mode train_ude --config configs/base_config.yaml --learning_rate 0.001 --epochs 200
```

## Примеры запуска на Kaggle

Для запуска кода на Kaggle, необходимо загрузить все файлы проекта и запустить через терминал:

```bash
cd /kaggle/working/
mkdir -p data models analysis configs
# Сначала копируем конфигурацию
cp /kaggle/input/base-config/base_config.yaml configs/
# Запускаем генерацию данных
python main.py --mode generate_data --config configs/base_config.yaml
# Запускаем обучение
python main.py --mode train_ude --config configs/base_config.yaml
# Запускаем анализ
python main.py --mode analyze_results --config configs/base_config.yaml
```

## Использование Weights & Biases (опционально)

Для логирования процесса обучения можно использовать Weights & Biases:

1. Установите библиотеку: `pip install wandb`
2. Включите логирование в конфигурации: `wandb.use_wandb: true`
3. Настройте свой проект: `wandb.project: "physics-ude"`
