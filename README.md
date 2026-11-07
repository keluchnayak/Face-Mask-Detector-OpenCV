# 😷 Face Mask Detector — OpenCV & Machine Learning Pipeline

## 📌 Project Overview
This project builds an end-to-end Computer Vision and Machine Learning application to automatically detect whether a person in an image is wearing a face mask (**`with_mask`** vs. **`without_mask`**).

The project is built on **Omkar Gurav's Face Mask Dataset** from Kaggle and strictly adheres to the 8-step image classification pipeline demonstrated in the OpenCV MNIST digit study.

---

## 📊 Dataset Specifications
* **Dataset Name:** Face Mask Dataset (by Omkar Gurav / Chandrika Deb)
* **Source Platform:** Kaggle
* **Classes:** 2 Categories (`with_mask`, `without_mask`)
* **Folder Structure:**
  ```text
  ├
  ├──Face Mask Detector.ipynb
  |
  |
  ├
  ├──data
  ├── with_mask       # Images of faces with protective masks
  └── without_mask    # Images of faces without masks
  ```

---

## ⚙️ 8-Step Machine Learning Pipeline

1. **Environment Setup & Imports**: Load OpenCV, NumPy, Pandas, Scikit-Learn, Pickle, and Gradio.
2. **Directory Traversal & Preprocessing**:
   * Read raw images in Grayscale mode: `cv2.imread(img_path, 0)`
   * Resize images to uniform spatial resolution: `cv2.resize(img, (28, 28))`
   * Flatten 2D image matrices into 1D vectors: `img.flatten()` (784 pixel feature values per image)
3. **Pandas Feature Matrix Construction**:
   * Build predictor matrix (`df1` with 784 pixel columns)
   * Create target label column (`df2`)
   * Concatenate into final dataset (`df`)
4. **Dataset Partitioning**: Perform an 80% Train / 20% Test stratified split using `train_test_split()`.
5. **Supervised Model Training**: Fit both **K-Nearest Neighbors (KNN)** and **Decision Tree** classifiers.
6. **Model Benchmarking**: Evaluate models using `accuracy_score()` and `classification_report()`.
7. **Model Persistence**: Serialize the selected trained model to `mask_model.pkl`.
8. **Inference & Web Deployment**: Create a standalone single-image `predict_mask()` function and deploy an interactive web interface using `gradio`.

---



## 🚀 Execution Instructions

1. **Download Dataset:** Download the **Face Mask Dataset** by Omkar Gurav from Kaggle.
2. **Directory Setup:** Unzip and place the dataset folder so that it contains the two class subfolders `data/with_mask/` and `data/without_mask/`.
3. **Run Notebook:** Execute Cells 1 through 8 sequentially in Jupyter Notebook or Google Colab.
4. **Interact:** Click the generated public link from `app.launch(share=True)` to open the web interface in your browser and test new images!
