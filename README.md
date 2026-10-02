# CV Buoys: Cardinal Mark Detection & Classification

> 🇫🇷 **Note on Language:** While this README and repository documentation are in English to accommodate international recruiters, please note that the underlying Python scripts, Jupyter Notebooks comments, and the original project report are written in **French**.

## 🎯 About The Project
The objective of this project is to develop algorithms to detect maritime cardinal buoys (identifying the bounding box coordinates) and classify their cardinal direction (West, North, South, East). 

Instead of relying on a single solution, this project is a comparative study of three distinct Computer Vision paradigms: Classical Image Analysis, Machine Learning, and Deep Learning.

## 🧠 Methodology

### 1. Classical Image Analysis
*   **Detection:** Uses HSV color space segmentation to extract yellow and dark hues, followed by morphological operations (opening and closing) to clean the binary masks and extract the bounding box.
*   **Classification:** Crops the buoys, applies horizontal dilation, and reads the sequence of yellow and black bands from top to bottom to deduce the cardinal direction.

### 2. Machine Learning
*   **Detection:** Relies on Histogram of Oriented Gradients (HOG) feature extraction combined with a Support Vector Machine (SVM) classifier, using a sliding window and bounding box fusion approach.
*   **Classification:** Uses K-Means clustering (initialized with two clusters) on color masks to find the highest concentrations of black and yellow pixels.

### 3. Deep Learning
*   **Detection:** Fine-tuning of the **YOLOv8n** model on a specifically formatted dataset (train/val split with `.yaml` configuration).
*   **Classification:** Fine-tuning of a pre-trained **ResNet-18** network with frozen final layers to classify the cropped bounding boxes into the four cardinal directions.

## 📊 Results & Performance Trade-offs

| Approach | Detection: Mean IoU | Detection: mAP @0.50 | Classification: Overall Accuracy |
| :--- | :---: | :---: | :---: |
| **Classical Vision** | 0.5991 | 0.6886 | to redo |
| **Machine Learning** | 0.5393 | 0.7024 | to redo |
| **Deep Learning** | **0.7009** | **0.7647** | **to redo** |

### 💡 Key Insights:
*   **Accuracy:** The Deep Learning approach (YOLOv8 + ResNet-18) provides the best overall scores, demonstrating excellent class separability.
*   **Computational Cost:** The Classical Image approach is the most efficient regarding CPU/GPU usage, relying on simple and lightweight operations. 
*   **Bottlenecks:** The ML approach using the SVM sliding window is computationally heavy, taking around 5 minutes to process the test images. Furthermore, hand-crafted classical approaches proved sensitive to environmental noise, such as the sea appearing dark or the sky mimicking yellow hues.

## 📁 Repository Structure

```text
cv-buoys/
│
├── src/                      # Underlying Python logic (French comments)
│   ├── classical_cv.py
│   ├── ml_pipeline.py
│   └── dl_pipeline.py
│
├── notebooks/                # Execution & Visualization (French)
│   ├── 01_classical_analysis.ipynb
│   ├── 02_machine_learning.ipynb
│   └── 03_deep_learning.ipynb
│
├── docs/
│   └── RAPPORT_IMLP_NGUYEN_PLAYE.pdf  # Original French Report
│
├── data/samples/             # Sample images for testing (Dataset excluded via .gitignore)
├── requirements.txt
└── README.md
