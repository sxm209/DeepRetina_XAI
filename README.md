# Retinal Disease Detection: Diabetic Retinopathy Classification

## Project Overview
This project leverages deep learning and computer vision to automatically detect and classify the severity of Diabetic Retinopathy (DR) from high-resolution retinal fundus images. Built using PyTorch and optimized for Apple Silicon (MPS), the model utilizes transfer learning with a ResNet50 architecture to evaluate images across a 0-4 severity scale. 

To bridge the gap between "black-box" AI and clinical utility, this project integrates Explainable AI (XAI) via **Grad-CAM** (Gradient-weighted Class Activation Mapping). This allows the model to visually highlight the specific anatomical regions—such as microaneurysms, hemorrhages, or exudates—that drove its predictive reasoning.

## Why This Project Matters
Diabetic Retinopathy is the leading cause of preventable blindness among working-age adults globally. Catching the disease early is critical, but screening requires specialized ophthalmologists to manually review fundus images, creating a massive bottleneck.

**The Value of this AI Solution:**
* **Scalability & Speed:** An automated model can process thousands of images in the time it takes a human specialist to review a dozen, acting as a high-throughput triage system.
* **Clinical Trust through XAI:** Doctors cannot blindly trust a neural network's diagnosis. By integrating Grad-CAM, the model provides visual evidence for its decisions, empowering clinicians to verify the AI's logic rather than just accepting a raw probability score. 
* **Compute Efficiency:** The pipeline is engineered to train and infer efficiently on modern hardware (like Apple's M1/M2/M3 architecture), proving that high-level medical imaging AI does not strictly require massive cloud-compute clusters to function effectively.

## Model Architecture & Advanced Techniques
To achieve robust performance on highly imbalanced medical data, this pipeline utilizes several advanced optimization strategies:
* **Targeted Fine-Tuning:** While leveraging a pretrained **ResNet50**, the final convolutional block (`layer4`) is unfrozen. This allows the network to learn domain-specific microscopic geometries (like neovascularization) while retaining fundamental edge detection from ImageNet.
* **Focal Loss:** The APTOS dataset is heavily skewed toward "Level 0" (Healthy). Standard loss functions allow the model to cheat by predicting the majority class. By implementing Focal Loss (`gamma=2`), the optimizer dynamically scales down the loss on easy examples, forcing the model to dedicate its learning capacity to the harder, rare, and more clinically dangerous cases (Levels 3 & 4).

## Dataset & Evaluation Strategy 
This project utilizes the **APTOS 2019 Blindness Detection** dataset. 

**A Note on the `test` directory:** Because this dataset was originally part of a competitive machine learning challenge, the hosts never publicly released the true labels for the test set to prevent cheating. Because we cannot evaluate our model's accuracy against unlabeled data, the provided `test` files are excluded from this training pipeline. Instead, we use robust validation techniques on the data we *do* have answers for:
1.  **Stratified Splitting:** We take the fully labeled `train.csv` and perform an 80/20 split to create our own localized Training and Validation sets.
2.  **Stratification** ensures that the class imbalance is proportionally maintained across both splits.
3.  We evaluate our model's actual performance solely on our 20% validation holdout, representing completely unseen data.

## Results & Clinical Viability
When deploying AI in a healthcare setting, raw accuracy is often a deceptive metric due to class imbalance. This project prioritized medical safety and recall over raw accuracy. 

* **Quadratic Weighted Kappa (QWK): 0.8281**
* **Validation Accuracy: 71%**

**Clinical Interpretation of Results:**
During optimization, a standard model achieved 77% accuracy by over-predicting the "Healthy" class, fatally missing over half of the most severe Level 4 cases. By introducing Focal Loss and fine-tuning, the overall accuracy shifted to 71%, but the model became significantly safer medically:
* **High Recall for Early Detection:** Recall for Level 1 (early-stage disease) jumped to 81%.
* **Minimized Catastrophic Error:** The outstanding QWK score of **0.8281** proves that when the model makes a misclassification, it is tightly bounded. It rarely confuses a severe case (Level 4) with a completely healthy eye (Level 0). In a medical triage environment, predicting one level adjacent to the ground truth is highly preferable to missing advanced disease entirely.

## Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch, Torchvision
* **Hardware Acceleration:** Metal Performance Shaders (MPS) for Apple Silicon
* **Computer Vision & Explainability:** OpenCV, pytorch-grad-cam
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Repository Structure
```text
DEEP_LEARNING_RETINA/
│
├── aptos2019-blindness-detection/   # Dataset directory (Ignored in version control)
│   ├── train_images/                # 3,662 high-res fundus images
│   ├── test_images/                 # Unlabeled competition images
│   ├── train.csv                    # Labels for train_images (0-4 severity)
│   ├── test.csv                     # Metadata for test_images
│   └── sample_submission.csv        
│
├── retina_model.ipynb               # Main Jupyter Notebook (Data loading, training, XAI)
└── README.md                        # Project documentation