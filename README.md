
# Medical Image Segmentation and Report Generation using UNet++, ResNet, and LLM

This repository provides an end-to-end pipeline for:
- Training a **UNet++ model** on medical images for segmentation and Legion Generation,
- Leveraging a **ResNet50 model** for feature extraction and Disease Prediction,
- And generating human-readable **LLM-based medical reports** from predictions.

---

## Project Structure

```
ML_PROJECT/
│
├── input/
│   └── PNG/
│       ├── Original/                # Original colonoscopy images
│       ├── Ground Truth/           # Corresponding segmentation masks
│       ├── sample_original/        # Sample images for quick testing
│       └── sample_ground_truth/    # Sample masks
│
├── models\logs/                    # Saved model checkpoints and logs
├── notebooks/                      # Jupyter notebooks for training and inference
├── pre_processing_data/           # Augmented or preprocessed datasets
├── report_outputs/                # Output text/image reports from LLM
├── source/                         # Utility functions and core code modules
│
├── config.yaml                     # Configuration file for model & paths
├── GroundTruth.csv                 # CSV for annotations or training metadata
├── predict.py                      # Script to run predictions
├── train.py                        # Script to train UNet++ model
├── requirements.txt                # All necessary dependencies
├── README.md                       # You're reading it!
```

---

## 🔧 Setup Instructions

### Python Version: `3.12` Recommended

### Create and Activate a Virtual Environment

#### Windows:
```bash
cd path\to\ML_PROJECT
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Linux/Mac:
```bash
cd /path/to/ML_PROJECT
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

##  Installation

Make sure to install all dependencies:
```bash
pip install -r requirements.txt
```

---

##  Training the Model

Train the UNet++ model using:
```bash
python train.py
```
Using the `notebooks/ResNet_Model_skin_disease.ipynb` train the ResNet Model 

Model checkpoints will be saved to: `models\logs/`

---

##  Predict with the Model

Run inference on a test image:
```bash
python predict.py --test_img input/sample_original/your_test_image.png
```
You can also use the Jupyter Notebooks for the predictions

Predicted masks will be saved alongside the image or in `report_outputs/`.

---

##  Report Generation with LLMs

After prediction, run the notebook to auto-generate reports:
- 📓 `notebooks/Connecting_ResNet_with_UNet_plus_and_LLM_report_generation.ipynb`

This notebook:
- Extracts visual features using ResNet50,
- Maps segmented regions to semantic classes,
- Uses an LLM (like OpenAI or HuggingFace) to generate a diagnostic report.

---

## 💾 Sample Data Structure

```
input/PNG/
├── Original/
│   └── 1.png, 2.png, ...
├── Ground Truth/
│   └── 1.png, 2.png, ...
```



## 📬 Contact

For questions or contributions, feel free to raise an issue or contact the maintainers:
1. yaseenah@buffalo.edu
2. mahmood7@buffalo.edu
3. kruthiks@buffalo.edu
---

