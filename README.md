# Brain Tumor Segmentation using U-Net

## Objective:
This project focuses on the segmentation of brain tumors from MRI scans using a deep learning approach. The goal is to accurately identify and segment tumor regions from multi-modal MRI data, which is critical for diagnosis, treatment planning, and monitoring of brain tumor patients.

## Key Features:

### Dataset:
- Utilized the BraTS dataset, which includes High-Grade Glioma (HGG) and Low-Grade Glioma (LGG) MRI scans.
- The dataset contains multi-modal MRI scans (FLAIR, T1, T1ce, T2) along with ground truth segmentation labels.

### Data Preprocessing:
- Loaded and preprocessed MRI scans using SimpleITK for efficient handling of medical imaging data.
- Cropped and resized images to focus on relevant regions, reducing unnecessary background information.
- Normalized the data and performed one-hot encoding on the ground truth labels for multi-class segmentation.

### Model Architecture:
- Implemented a U-Net model, a state-of-the-art architecture for biomedical image segmentation.
- The U-Net consists of an encoder-decoder structure with skip connections to preserve spatial information.
- Used Batch Normalization and Dropout layers to improve model generalization and prevent overfitting.

### Loss Function:
- Employed Dice Loss as the primary loss function, which is well-suited for segmentation tasks due to its ability to handle class imbalance.
- Dice Coefficient was used as the evaluation metric to measure the overlap between predicted and ground truth segmentation masks.

### Training:
- Trained the model using the Adam optimizer with a learning rate of 1e-5.
- Implemented early stopping and model checkpointing to save the best-performing model during training.
- Achieved a high Dice Coefficient of **0.9950** on the validation set, demonstrating the model's effectiveness.

### Evaluation:
- Evaluated the model on separate test sets for both HGG and LGG cases.
- Achieved Dice Coefficients of **0.9795**, **0.9855**, and **0.9793** on HGG test sets, and **0.9950** on the LGG test set.
- Visualized the segmentation results by comparing predicted masks with ground truth labels, showing strong alignment.

### Tools and Libraries:
- **Python Libraries:** TensorFlow, Keras, SimpleITK, NumPy, Matplotlib, scikit-learn.

## Results:
The model successfully segmented brain tumors with high accuracy, as evidenced by the Dice Coefficient scores and visual comparisons between predicted and ground truth masks. The U-Net architecture, combined with Dice Loss, proved to be highly effective for this task, particularly in handling the complex and varied structures of brain tumors.

## Future Work:
- Explore additional data augmentation techniques to further improve model robustness.
- Experiment with other loss functions, such as Focal Loss or Tversky Loss, to address class imbalance.
- Extend the model to 3D segmentation for more comprehensive analysis of volumetric MRI data.

