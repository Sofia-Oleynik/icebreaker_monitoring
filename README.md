# Iceberg Detection and Monitoring System in the Barents Sea Using Remote Sensing and Machine Learning

This project is a software system for automatic detection and classification of icebergs and ships in the Barents Sea using satellite radar data (Sentinel-1) and deep learning methods. The system processes Synthetic Aperture Radar (SAR) images, applies preprocessing techniques, and uses a convolutional neural network (CNN) to distinguish between icebergs and ships.

## Project Description

Monitoring icebergs in the Barents Sea is crucial for maritime safety, offshore operations, and climate research. Traditional methods of iceberg detection are labor-intensive and often unreliable in polar regions due to cloud cover and limited daylight. This system addresses these challenges by utilizing freely available Sentinel-1 SAR data, which can penetrate clouds and operate day and night.

The workflow consists of three main stages:

1. **Data Preprocessing (SNAP):** The system uses the ESA SNAP toolbox to process raw SAR data. Key preprocessing steps include:
   - Orbit file application for precise geolocation
   - Radiometric calibration to convert digital numbers to backscatter coefficients (sigma nought)
   - Terrain correction using a Digital Elevation Model (DEM) to account for geometric distortions
   - Conversion to decibels (dB) for improved visualization
   - Land masking using a shapefile to remove land areas from consideration
   - Subsetting to focus on the region of interest

2. **Data Preparation:** After preprocessing, the following bands are extracted:
   - **HH (Horizontal-Horizontal):** Co-polarized backscatter
   - **HV (Horizontal-Vertical):** Cross-polarized backscatter
   - **Total Backscatter (b3):** HH + HV
   - **Cross Polarization Ratio (c3):** HV / (HH + HV)
   
   The data is then normalized using incidence angle correction and Lee filter smoothing to reduce speckle noise.

3. **Object Detection (CNN):** A convolutional neural network model classifies detected objects into three categories:
   - **Ship** (red bounding boxes)
   - **Iceberg** (blue bounding boxes)
   - **Unidentified Floating Object** (purple bounding boxes)

## Detection Workflow

1. **Import SAR Data:** The system loads a Sentinel-1 GRDH product in `.zip` format.
2. **Apply Orbit File:** Precise orbit information is applied to improve geolocation accuracy.
3. **Calibration:** Radiometric calibration converts the data to sigma nought backscatter values.
4. **Terrain Correction:** The data is projected to a map coordinate system (UTM Zone 33) with 10m pixel spacing.
5. **Convert to dB:** Linear backscatter is converted to decibels for better contrast.
6. **Land Masking:** A shapefile is used to mask out land areas, focusing only on the ocean.
7. **Subsetting:** The scene is cropped to a region of interest (e.g., 1000x1000 pixels).
8. **Band Extraction:** HH, HV, incidence angle, latitude, and longitude bands are extracted.
9. **Potential Object Detection:** A threshold is applied to the total backscatter band to identify potential objects (ROIs).
10. **Object Classification:** For each detected object, a 75x75 pixel subset is extracted and fed into the CNN model for classification.

## Technologies

- **Programming Languages:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **SAR Data Processing:** ESA SNAP (via `snappy` Python interface)
- **Image Processing:** Pillow, scikit-image, scipy
- **Data Manipulation:** NumPy, pandas
- **Visualization:** Matplotlib
- **SAR Data Source:** Sentinel-1 (ESA)
- **Model Architecture:** Custom Convolutional Neural Network (CNN)

## Installation

### 1. Prerequisites

- Python 3.7 or higher
- ESA SNAP with Python interface (`snappy`)

### 2. Clone the Repository

```bash
git clone https://github.com/your_username/iceberg-detection-barents-sea.git
cd iceberg-detection-barents-sea
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Land Mask Shapefile

Download the ocean shapefile from [Natural Earth](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-ocean/) and place it in the `ne_10m_ocean/` directory.

### 5. Run the System

```bash
python main.py
```

## Model Architecture

The CNN model uses the following architecture:

```python
Input (75x75x4)
    ↓
Conv2D(16, 3x3) + ReLU → MaxPooling2D(2x2) → Dropout(0.1)
    ↓
Conv2D(32, 3x3) + ReLU → MaxPooling2D(2x2) → Dropout(0.1)
    ↓
Conv2D(64, 3x3) + ReLU → MaxPooling2D(2x2) → Dropout(0.1)
    ↓
Conv2D(128, 3x3) + ReLU → Conv2D(128, 3x3) + ReLU → MaxPooling2D(2x2) → Dropout(0.1)
    ↓
Flatten
    ↓
Dense(64) + ReLU → Dense(64) + ReLU → Dense(64) + ReLU
    ↓
Dense(1) + Sigmoid (Output)
```

## Dataset

The model was trained on a dataset of SAR images with labeled examples of icebergs and ships. The dataset includes:

- **HH (band_1):** Co-polarized backscatter (75x75 pixels)
- **HV (band_2):** Cross-polarized backscatter (75x75 pixels)
- **Incidence Angle:** Local incidence angle for normalization
- **Labels:** Binary classification (iceberg vs. ship)

### Data Augmentation

To improve model generalization, the following augmentations are applied:
- Random rotation (up to 90 degrees)
- Width and height shifts (20%)
- Shear transformations (20%)
- Zoom (20%)
- Horizontal flipping

## Results

The model achieves the following performance metrics on the validation set:

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | ~88% |
| **Validation Loss** | ~0.30 |
| **Test Accuracy** | ~85-88% |

### Training Curves

The model was trained for 100 epochs with early stopping (patience=50). The training and validation curves show stable convergence.

<img width="1041" height="804" alt="Training and validation loss curves" src="https://github.com/user-attachments/assets/b58fd5bc-3003-4d0a-ae04-a30333867e38" />
*Figure 1: Training and validation loss over 100 epochs.*

### Detection Results

<img width="948" height="644" alt="Detection results with classified objects" src="https://github.com/user-attachments/assets/a111cc98-3e57-420b-9ae6-fe6476e3e06c" />
*Figure 2: Detection results showing ships (red), icebergs (blue), and unidentified objects (purple) with confidence scores.*

## Output

The system generates the following outputs:

1. **Visualization Plot:** A plot showing the classified objects with bounding boxes and confidence scores.
2. **CSV File:** A table containing the following columns for each detected object:
   - Latitude
   - Longitude
   - Object Type (Ship / Iceberg / Unidentified)
   - Probability (confidence score)

## Future Development

- Integration of additional SAR bands (e.g., VV, VH)
- Implementation of multi-temporal analysis for tracking iceberg movement
- Deployment as a web-based monitoring system
- Integration with AIS data for improved ship classification
- Real-time processing pipeline

---------------------------------------

# Система обнаружения и мониторинга айсбергов в Баренцевом море с использованием данных ДЗЗ и методов машинного обучения

Этот проект представляет собой программную систему для автоматического обнаружения и классификации айсбергов и судов в Баренцевом море с использованием спутниковых радиолокационных данных (Sentinel-1) и методов глубокого обучения. Система обрабатывает изображения радиолокаторов с синтезированной апертурой (РСА), применяет методы предварительной обработки и использует сверточную нейронную сеть (CNN) для различения айсбергов и судов.

## Описание проекта

Мониторинг айсбергов в Баренцевом море имеет решающее значение для безопасности мореплавания, морских операций и климатических исследований. Традиционные методы обнаружения айсбергов трудоемки и часто ненадежны в полярных регионах из-за облачного покрова и ограниченного светового дня. Данная система решает эти проблемы, используя общедоступные данные РСА Sentinel-1, которые могут проникать сквозь облака и работать круглосуточно.

Рабочий процесс состоит из трех основных этапов:

1.  **Предварительная обработка данных (SNAP):** Система использует инструментарий ESA SNAP для обработки сырых РСА-данных. Ключевые этапы предварительной обработки включают:
    - Применение файла орбиты для точного геопозиционирования
    - Радиометрическую калибровку для преобразования цифровых значений в коэффициенты обратного рассеяния (сигма-ноль)
    - Коррекцию рельефа с использованием цифровой модели рельефа (ЦМР) для учета геометрических искажений
    - Преобразование в децибелы (дБ) для улучшения визуализации
    - Маскирование суши с использованием шейп-файла для исключения наземных участков
    - Вырезание подмножества для фокусировки на области интереса

2.  **Подготовка данных:** После предварительной обработки извлекаются следующие каналы:
    - **HH (горизонтальная-горизонтальная):** Со-поляризованное обратное рассеяние
    - **HV (горизонтальная-вертикальная):** Кросс-поляризованное обратное рассеяние
    - **Суммарное обратное рассеяние (b3):** HH + HV
    - **Коэффициент кросс-поляризации (c3):** HV / (HH + HV)
    
    Затем данные нормализуются с использованием коррекции угла падения и фильтрации Ли для уменьшения спекл-шума.

3.  **Обнаружение объектов (CNN):** Модель сверточной нейронной сети классифицирует обнаруженные объекты на три категории:
    - **Судно** (красные ограничивающие рамки)
    - **Айсберг** (синие ограничивающие рамки)
    - **Неопознанный плавающий объект** (фиолетовые ограничивающие рамки)

## Рабочий процесс обнаружения

1.  **Импорт РСА-данных:** Система загружает продукт Sentinel-1 GRDH в формате `.zip`.
2.  **Применение файла орбиты:** Применяется точная орбитальная информация для улучшения точности геопозиционирования.
3.  **Калибровка:** Радиометрическая калибровка преобразует данные в значения обратного рассеяния сигма-ноль.
4.  **Коррекция рельефа:** Данные проецируются в картографическую систему координат (UTM зона 33) с пространственным разрешением 10 м.
5.  **Преобразование в дБ:** Линейное обратное рассеяние преобразуется в децибелы для лучшего контраста.
6.  **Маскирование суши:** Шейп-файл используется для маскирования наземных участков, фокусируясь только на океане.
7.  **Вырезание подмножества:** Сцена обрезается до области интереса (например, 1000x1000 пикселей).
8.  **Извлечение каналов:** Извлекаются каналы HH, HV, угол падения, широта и долгота.
9.  **Обнаружение потенциальных объектов:** К каналу суммарного обратного рассеяния применяется порог для выявления потенциальных объектов (областей интереса).
10. **Классификация объектов:** Для каждого обнаруженного объекта извлекается подмножество размером 75x75 пикселей и подается в модель CNN для классификации.

## Стек технологий

- **Языки программирования:** Python
- **Фреймворк глубокого обучения:** TensorFlow / Keras
- **Обработка РСА-данных:** ESA SNAP (через Python-интерфейс `snappy`)
- **Обработка изображений:** Pillow, scikit-image, scipy
- **Обработка данных:** NumPy, pandas
- **Визуализация:** Matplotlib
- **Источник РСА-данных:** Sentinel-1 (ESA)
- **Архитектура модели:** Сверточная нейронная сеть (CNN)

## Установка

### 1. Предварительные требования

- Python 3.7 или выше
- ESA SNAP с Python-интерфейсом (`snappy`)

### 2. Клонирование репозитория

```bash
git clone https://github.com/your_username/iceberg-detection-barents-sea.git
cd iceberg-detection-barents-sea
```

### 3. Установка зависимостей Python

```bash
pip install -r requirements.txt
```

### 4. Загрузка шейп-файла для маскирования суши

Загрузите шейп-файл океана с сайта [Natural Earth](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-ocean/) и поместите его в директорию `ne_10m_ocean/`.

### 5. Запуск системы

```bash
python main.py
```

## Архитектура модели

Модель CNN использует следующую архитектуру:

```python
Вход (75x75x4)
    ↓
Conv2D(16, 3x3) + ReLU → MaxPooling2D(2x2) → Dropout(0.1)
    ↓
Conv2D(32, 3x3) + ReLU → MaxPooling2D(2x2) → Dropout(0.1)
    ↓
Conv2D(64, 3x3) + ReLU → MaxPooling2D(2x2) → Dropout(0.1)
    ↓
Conv2D(128, 3x3) + ReLU → Conv2D(128, 3x3) + ReLU → MaxPooling2D(2x2) → Dropout(0.1)
    ↓
Flatten
    ↓
Dense(64) + ReLU → Dense(64) + ReLU → Dense(64) + ReLU
    ↓
Dense(1) + Sigmoid (Выход)
```

## Набор данных

Модель была обучена на наборе РСА-изображений с размеченными примерами айсбергов и судов. Набор данных включает:

- **HH (band_1):** Со-поляризованное обратное рассеяние (75x75 пикселей)
- **HV (band_2):** Кросс-поляризованное обратное рассеяние (75x75 пикселей)
- **Угол падения:** Локальный угол падения для нормализации
- **Метки:** Бинарная классификация (айсберг vs. судно)

### Аугментация данных

Для улучшения обобщения модели применяются следующие аугментации:
- Случайное вращение (до 90 градусов)
- Сдвиг по ширине и высоте (20%)
- Преобразования сдвига (20%)
- Масштабирование (20%)
- Горизонтальное отражение

## Результаты

Модель достигает следующих показателей производительности на валидационном наборе:

| Метрика | Значение |
|---------|----------|
| **Точность на валидации** | ~88% |
| **Потери на валидации** | ~0.30 |
| **Точность на тесте** | ~85-88% |

### Кривые обучения

Модель обучалась в течение 100 эпох с ранней остановкой (patience=50). Кривые обучения и валидации показывают стабильную сходимость.

<img width="1041" height="804" alt="loss" src="https://github.com/user-attachments/assets/b58fd5bc-3003-4d0a-ae04-a30333867e38" />

### Результаты обнаружения

<img width="948" height="644" alt="detection" src="https://github.com/user-attachments/assets/a111cc98-3e57-420b-9ae6-fe6476e3e06c" />

## Выходные данные

Система генерирует следующие выходные данные:

1.  **Визуализационный график:** График с классифицированными объектами, ограничивающими рамками и значениями уверенности.
2.  **CSV-файл:** Таблица со следующими колонками для каждого обнаруженного объекта:
    - Широта
    - Долгота
    - Тип объекта (Судно / Айсберг / Неопознано)
    - Вероятность (значение уверенности)

## Перспективы развития

- Интеграция дополнительных РСА-каналов (например, VV, VH)
- Реализация многомерного анализа для отслеживания движения айсбергов
- Развертывание в виде веб-системы мониторинга
- Интеграция с данными AIS для улучшения классификации судов
- Создание конвейера обработки в реальном времени
