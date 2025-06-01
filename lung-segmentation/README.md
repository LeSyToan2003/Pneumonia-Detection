#   U-Net Showdown —  Comparing Architectures for Convex Hull on Lung Region Segmentation

Link for Datasets: https://www.kaggle.com/datasets/yunannnn/lung-segmentation-dataset-ch0

## Project Information

Lung region segmentation is a crucial step in medical image processing, particularly in disease diagnosis based on chest X-ray images. In this study, we focus on evaluating and comparing the performance of different U-Net variants - one of the leading architectures for image segmentation - in the task of lung region extraction.

The models under comparison include the original U-Net, U-Net++, Attention U-Net and R2U-Net, with performance assessed based on metrics such as Dice Score, Intersection over Union (IoU), and inference time. The study utilizes the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and [COVID-QU-Ex Dataset](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu), these are the high-quality collections of chest X-ray images. The effectiveness of the models is validated on the [Pulmonary Chest X-Ray Dataset](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)

The results of this research will highlight the strengths and weaknesses of each model, providing valuable insights for clinical applications, data preprocessing, and the enhancement of automated diagnosis systems. 

<center> <img src="image_1.png" alt="alt text" width="800"> </center>

By applying Convex Hull to the segmented mask, we ensure that small gaps and irregularities are smoothed out, leading to a more complete and continuous region. This is particularly beneficial in lung segmentation from chest X-ray images, where deep learning models may sometimes fail to capture the full lung boundary due to low contrast or occlusions.

From a metric perspective, using Convex Hull can significantly improve recall, as it helps recover missing parts of the predicted segmentation. Recall is defined as:

$$
    Recall = \frac{TP}{TP+FN}
$$

Since Convex Hull reduces false negatives (FN) by expanding the predicted region to cover more of the actual object, it results in a higher recall score. This is especially useful in medical applications where under-segmentation (missing important structures) can lead to misdiagnosis or incomplete analysis.

<center> <img src="image_3.png" alt="alt text" width="800"> </center>

A higher recall directly benefits disease diagnosis models by minimizing the risk of missing critical pathological areas. This is especially important in conditions like pneumonia or lung cancer, where an incomplete segmentation could lead to inaccurate feature extraction and misdiagnosis. By enhancing recall, we provide a more comprehensive input for classification models, ultimately improving their reliability and clinical applicability.

**Environment: [Kaggle](https://www.kaggle.com/)**

## Libraries:

- numpy
- matplotlib
- cv2 
- sklearn
- tensorflow

## Models:

- **U-Net**: https://arxiv.org/abs/1505.04597
- **Attention U-Net**: https://arxiv.org/abs/1804.03999
- **U-Net++**: https://arxiv.org/abs/1807.10165
- **R2U-Net**: https://arxiv.org/abs/1802.06955

**Open Sources for Models: https://github.com/yingkaisha/keras-unet-collection**

Models are trained on **GPU P100**.

## Algorithms: Convex Hull of Points

- **Graham Scan**: https://doi.org/10.1016/0020-0190(73)90020-3
- **Andrew’s Monotone Chain**: https://doi.org/10.1016/0020-0190(79)90072-3