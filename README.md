# MLOps Cancer Classification

## Overview  
Image‑based cancer classifier that processes input images through a pipeline including preprocessing, model inference, and a REST API interface.

## Structure  

- `src/CancerClassifier/`: core model inference & utility code  
- `research/`: experiments, notebooks, prototyping  
- `model/`: trained models / saved artifacts  
- `config/`: configuration files (`params.yaml`, etc.)  
- `app.py`, `main.py`: web/API endpoints for inference  
- `templates/`, `index.html`: frontend/interface for uploading/viewing results  
- DVC setup (`dvc.yaml`, `.dvc`, `.dvcignore`): data & experiment versioning  

## Setup  

```bash
git clone https://github.com/SA-Duran/MLOps_cancer_classification.git
cd MLOps_cancer_classification
pip install -r requirements.txt
```

## Inference API  

```bash
python app.py
```

Then use the web interface (`index.html`) or send requests to the API to classify images (upload or send image input).  

## Training / Experimentation  

Configuration driven via `params.yaml` and DVC. Use notebook(s) in `research/` to explore, train, and evaluate.

## Evaluation & Outputs  

- Model performance stored in `scores.json`  
- Input sample image is `inputImage.jpg`  
- Templates show visual feedback in the UI  

## Tools  

- Python  
- DVC (Data Version Control)  
- Flask (or similar) for REST / web interface  
- Jupyter notebooks  
- HTML templates  

## Next Steps  

- Add model explainability (e.g. heatmaps)  
- Improve front‑end UX  
- Add more metrics & confidence reporting  
- Automate CI/CD / deployment  

## License  
MIT (add or verify `LICENSE` file)
