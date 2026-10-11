# Military Object Detection

Object detection for military vehicles and personnel in photographs and drone-style imagery. The project fine-tunes YOLOv8 on a COCO/YOLO-format dataset of five classes, and ships a Streamlit demo and a FastAPI inference service.

## Classes

`ARMOREED_CAR`, `ICV`, `PERSON`, `TANK`, `TRUCK`

The spelling of `ARMOREED_CAR` follows the label name in the source dataset export. A `CIVILIAN_CAR` class present in the original data was removed during preprocessing so the model does not act as a general vehicle detector.

## Results

Final model: YOLOv8m, 50 epochs, 640 px input, with strong augmentation (mosaic, mixup, HSV jitter, perspective, random erasing). Evaluated on the validation split (1992 images, 1812 instances):

| Class        | Precision | Recall | mAP@50 | mAP@50-95 |
|--------------|-----------|--------|--------|-----------|
| ARMOREED_CAR | 0.922     | 0.635  | 0.818  | 0.554     |
| ICV          | 0.983     | 0.899  | 0.951  | 0.713     |
| PERSON       | 0.848     | 0.614  | 0.749  | 0.390     |
| TANK         | 0.962     | 0.907  | 0.949  | 0.648     |
| TRUCK        | 0.922     | 0.861  | 0.923  | 0.630     |
| **All**      | 0.927     | 0.783  | 0.878  | 0.587     |

An earlier YOLOv8n baseline (50 epochs, default augmentation) is kept in [results/yolov8n_baseline](results/yolov8n_baseline) with its training log and hyperparameters. It reached 0.894 mAP@50 and 0.606 mAP@50-95, but it was trained on a different label set (before the civilian class was removed), so the two sets of numbers are not directly comparable.

Recall on `PERSON` and `ARMOREED_CAR` is the weakest point. Persons are small and numerous in the data, and armored cars are the least represented vehicle class.

## Repository layout

```
app/              Streamlit demo
deployment/       FastAPI service, Dockerfile and its requirements
docs/             Project report, slides, brief and LaTeX source
notebooks/        EDA and training notebooks (designed for Google Colab)
results/          Training logs and hyperparameters of the baseline run
scripts/          Helper scripts (weight download)
src/              Dataset, preprocessing, metrics and visualization utilities
tests/            API tests
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Class balance, bounding box statistics, spatial heatmaps, annotation checks |
| `02_train_faster_rcnn.ipynb` | Faster R-CNN training template (torchvision) |
| `03_train_yolo.ipynb` | YOLOv8 baseline training template |
| `04_yolov8m_training.ipynb` | Final YOLOv8m pipeline: label cleaning, heavy augmentation, evaluation |

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/gabrielyannarek04-max/military-object-detection.git
cd military-object-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Weights

Trained weights are not stored in the repository. Download them from the project's GitHub Releases page into `models/best.pt`:

```bash
python scripts/download_weights.py
```

Alternatively, place your own YOLOv8 weights at `models/best.pt`, or point the `MODEL_PATH` environment variable at them.

## Usage

### Demo application

```bash
streamlit run app/streamlit_app.py
```

Upload an image and adjust the confidence and NMS IoU thresholds from the sidebar.

### REST API

```bash
pip install -r deployment/requirements.txt
cd deployment
uvicorn main:app --port 8000
```

Request:

```bash
curl -X POST http://localhost:8000/predict/ -F "file=@image.jpg"
```

Response:

```json
{
  "status": "success",
  "predictions": [
    {"class_id": 3, "class_name": "TANK", "confidence": 0.91, "bbox": [112.4, 60.0, 340.8, 215.2]}
  ]
}
```

`bbox` is `[xmin, ymin, xmax, ymax]` in pixels. An optional `conf` query parameter sets the confidence threshold (default 0.25). `GET /health` reports service status and whether weights were found. If weights are missing, `/predict/` returns HTTP 503.

### Docker

```bash
docker build -f deployment/Dockerfile -t military-detection-api .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" military-detection-api
```

The image is CPU-only. For GPU inference, use a CUDA-enabled PyTorch base image.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a stub model, so neither weights nor a GPU are required.

## Limitations

- The dataset is small and drawn from a limited set of sources. Performance on imagery with different sensors, altitudes or camouflage is not measured.
- Validation numbers come from a single split; no separate held-out test set was evaluated.
- This is an academic project and should not be used for operational decisions.

## Documentation

The full write-up is in [docs/report.pdf](docs/report.pdf), with slides in [docs/slides.pdf](docs/slides.pdf).

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Narek Gabrielyan.
