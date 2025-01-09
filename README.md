# Object Detection on xView and SkyFusion Datasets

This project represents our effort in addressing the challenging task of object detection on large-scale datasets. My name is Leonardo Catello, and alongside my colleagues Sara Meglio and Aurora D’Ambrosio, we explored various models and methodologies to detect objects in satellite imagery. Our work primarily focused on the **xView dataset**, with a final application to the **SkyFusion dataset** to validate our findings.

We structured this journey by initially diving into the characteristics of the dataset and its preprocessing. Then, we implemented a series of object detection models, ranging from the traditional R-CNN to advanced frameworks like Faster R-CNN, SSD, and YOLO, analyzing their performance and limitations. Here's a detailed walkthrough of our approach and results.

---

## Dataset and Preprocessing

### Exploring the xView Dataset

The **xView dataset** is widely regarded as one of the most comprehensive resources for satellite imagery object detection. It spans a remarkable area of **1400 km²**, with a resolution of **0.3 meters per pixel**, meaning each pixel represents just 30 centimeters on the ground. This fine granularity allows for precise detection of objects.

The dataset consists of:
- **60 object classes**, ranging from vehicles and buildings to helipads and storage tanks.
- Nearly **1 million labeled objects** across **1127 high-resolution images**.
- A staggering **33.54 GB** of data.

These characteristics make xView both an exciting and demanding dataset for deep learning applications. 

### Preprocessing Challenges and Solutions

Working with xView presented several challenges due to its size and complexity. To adapt it to our computational resources and the requirements of various models, we performed extensive preprocessing:

1. **Cleaning the Dataset**  
   We started by addressing inconsistencies, such as missing or corrupted images. Annotation files in the original **GeoJSON** format were converted to the more standardized **COCO JSON**, which is widely supported by modern object detection frameworks. Images were also transformed from the TIFF format to **JPG** to optimize storage and processing.

2. **Dimensionality Reduction**  
   The large size of the images posed challenges for model training. To resolve this, we divided each image into smaller **320x320 pixel chunks**, ensuring each chunk was easier to process. Moreover, **67% of empty chunks** (those without labeled objects) were removed to focus computational resources on meaningful data.

3. **Category Aggregation**  
   While the dataset originally featured 60 classes, many were too fine-grained or underrepresented. To streamline analysis and enhance model performance, we grouped these into **11 macro-classes**, such as Aircraft, Vehicles, Buildings, and Storage Tanks, plus a **background** class.

### The Final Dataset

After preprocessing, our dataset was significantly more manageable:
- **670,000 labeled objects**  
- **45,891 image chunks**, of which **30%** were empty  
- Overall size reduced to **1.69 GB**

Despite these improvements, the dataset remained imbalanced, with some classes, such as "Building," dominating the distribution. This imbalance required targeted strategies during training, such as undersampling and oversampling, to mitigate its impact.

---

## R-CNN: A Pipeline Approach to Object Detection

Our first model was the **Region-based Convolutional Neural Network (R-CNN)**, a pioneering framework that divides the object detection process into distinct steps. This pipeline involves generating region proposals, extracting features for classification, and refining bounding box coordinates. While computationally intensive, R-CNN provided a clear starting point for tackling the xView dataset.

### Region Proposals with Selective Search

The initial step in R-CNN is generating candidate regions likely to contain objects. We used the **Selective Search** algorithm, which segments images into smaller groups of similar pixels, known as superpixels. These are iteratively merged based on texture, color, and size, resulting in a set of proposed regions.

To optimize this process, we excluded regions that were excessively small, large, or skewed. Images were resized to **160x160 pixels** during this stage, and **batch processing** was implemented to accelerate computations.

### Classification and Refinement

Each proposed region was processed using CNNs for feature extraction and classification:
- **AlexNet** was our first choice due to its simplicity and speed. Despite extensive tuning, its performance was unsatisfactory, with an accuracy of only **0.52%**. The model struggled with imbalances, often favoring dominant classes.
- **ResNet50**, a more advanced architecture, significantly improved results, achieving an accuracy of **62%**. It leveraged deeper feature extraction to better differentiate between classes.

### Bounding Box Regressor: Refining Object Localization

One of the key challenges in the R-CNN pipeline lies in the **region proposals** generated during the initial stage. These proposals often fail to perfectly align with the objects they aim to capture. They might be too large, too small, or poorly positioned, with edges that either cut off parts of the object or extend unnecessarily into the background. To address this, the **Bounding Box Regressor** was implemented as a critical step to refine these proposals and improve localization accuracy.

#### How the Bounding Box Regressor Works

The goal of the bounding box regressor is to adjust an initial proposal \(P\) (the predicted bounding box) so that it more accurately matches the ground truth box \(G\). This is achieved by learning a transformation that aligns the coordinates and dimensions of \(P\) with those of \(G\). The transformation uses four key adjustments:
1. **Shifting the box's position** to better center it on the object.
2. **Scaling its dimensions** to match the true size of the object.

Mathematically, the adjustments involve both linear and logarithmic transformations. These transformations refine the x and y coordinates, as well as the width (\(w\)) and height (\(h\)) of the bounding box:

\[
\hat{G}_x = P_w \hat{t}_x + P_x, \quad \hat{G}_y = P_h \hat{t}_y + P_y, \quad
\hat{G}_w = P_w \exp(\hat{t}_w), \quad \hat{G}_h = P_h \exp(\hat{t}_h)
\]

Here, \(\hat{t}_x, \hat{t}_y, \hat{t}_w, \hat{t}_h\) represent the adjustments needed to transform \(P\) into \(\hat{G}\), the refined bounding box. Conversely, the regression targets \(t_x, t_y, t_w, t_h\) are calculated during training as:

\[
t_x = \frac{G_x - P_x}{P_w}, \quad t_y = \frac{G_y - P_y}{P_h}, \quad
t_w = \log \left( \frac{G_w}{P_w} \right), \quad
t_h = \log \left( \frac{G_h}{P_h} \right)
\]

These targets are normalized to ensure stability during training, particularly for the positional offsets (\(t_x, t_y\)).

#### Training the Regressor

The regressor is trained using a **regularized linear regression** model, which computes separate weights for each transformation parameter (\(x, y, w, h\)) for every class. The regression weights are derived by solving the following equation:

\[
w_k = (P^T P + \lambda I)^{-1} P^T t_k
\]

Here:
- \(P\) represents the coordinates of the proposed bounding boxes.
- \(t_k\) is the regression target for the corresponding parameter (\(k \in \{x, y, w, h\}\)).
- \(\lambda\) is the regularization term, which prevents overfitting by penalizing large weights.

The regressor is trained globally across all training data for a given class. This approach generalizes the adjustments needed for any bounding box of that class, ensuring robust corrections during inference.

#### Results and Challenges

The bounding box regressor significantly improved localization by refining proposals to better align with the ground truth. However, its effectiveness was still constrained by the quality of the initial region proposals generated by **Selective Search**. 

As shown below, while the regressor reduced errors in box placement, the total number of predicted boxes fell short compared to the ground truth:

![Bounding Box Refinement Example](https://prod-files-secure.s3.us-west-2.amazonaws.com/fa1b5077-0ed9-44f4-b1ae-33f8b9441bfb/a849dc38-4781-447e-a716-95d7861939a4/immagine_(1).png)

This limitation stems from the inability of Selective Search to generate enough high-quality proposals, particularly under the resource constraints of this project. Despite the regressor's improvements, the model's overall performance suffered due to insufficient region coverage, emphasizing the need for more advanced region proposal methods, such as those integrated in Faster R-CNN.

--- 

## Dataset Preparation: Undersampling and Oversampling

Given the significant imbalance in the xView dataset, we employed a combination of **undersampling** and **oversampling** to balance the categories for training our models. The preprocessing decisions were crucial for improving the model’s performance across all classes.

For **undersampling**, we removed images where over 90% of the objects belonged to the dominant "Building" class. Unlike in R-CNN, where we could target specific bounding boxes, working with the raw dataset required us to eliminate entire images. While this approach reduced the dominance of the "Building" category, it preserved sufficient data for other classes. 

However, undersampling alone left the dataset still imbalanced. To address this, we applied **oversampling**, duplicating images and their annotations to increase the representation of underrepresented classes. To avoid inflating unrelated categories, we selected images with the highest occurrence of target class bounding boxes. Through this method, classes like `[2, 3, 4, 5, 7, 8, 9, 10, 11]` saw a significant boost, while class `[1]` was increased moderately.

Initially, the dataset contained **50,416 objects across 4,589 images**. After undersampling, these numbers dropped to **26,422 objects in 3,304 images**. Finally, oversampling balanced the dataset at **48,993 objects distributed over 6,092 images**, ensuring a more even distribution of class instances. 

Before this process, the class distribution was heavily skewed, as illustrated in the "Before" chart below. After our interventions, the "After" chart demonstrates a significantly more balanced dataset, creating a robust foundation for training.

**Before Balancing:**

![Before](https://prod-files-secure.s3.us-west-2.amazonaws.com/fa1b5077-0ed9-44f4-b1ae-33f8b9441bfb/18889727-56e1-468f-ad55-fe7c4bba0cdb/d1822530-5751-41ef-a931-44822d80f34a.png)

**After Balancing:**

![After](https://prod-files-secure.s3.us-west-2.amazonaws.com/fa1b5077-0ed9-44f4-b1ae-33f8b9441bfb/69152414-19b7-4c95-b5cb-d6f7f5a05c1f/9b9e9880-d0dd-48a0-b669-bbf7b76ed16a.png)

---

## Faster R-CNN: Region Proposal and End-to-End Training

Unlike R-CNN, where region proposals were generated using an external algorithm like Selective Search, Faster R-CNN incorporates a **Region Proposal Network (RPN)** directly into its architecture. This innovation eliminates the need for a separate proposal generation step, as the RPN leverages feature maps produced by the CNN to propose regions of interest (RoIs).

Faster R-CNN optimizes the entire detection pipeline through four joint objectives:
1. **RPN Classification**: Classifying proposed regions as either "object" or "background."
2. **RPN Box Regression**: Refining bounding box coordinates for proposed regions.
3. **Final Classification**: Determining the class of the detected object.
4. **Final Box Regression**: Further adjusting bounding box coordinates for final predictions.

For this study, we used a Faster R-CNN model with a **ResNet50 backbone** and a **Feature Pyramid Network (FPN)**. The FPN plays a critical role in handling objects at varying scales, a common challenge in satellite imagery. By creating a multiscale feature pyramid, the FPN enhances the model's ability to detect both large and small objects effectively.

Given the computational complexity of the model, we limited training to **10 epochs** and utilized only **10% of the dataset**. The data was split in a stratified manner: 80% for training, 10% for validation, and 10% for testing.

### Training Results and Metrics

The training process demonstrated significant improvements in key metrics:
- **Precision** increased steadily, indicating a reduction in false positives. Training precision reached values near **0.7**, while validation precision hovered around **0.5–0.6**, albeit with some fluctuations.
- **Recall** remained consistently high, exceeding **96%** for validation and nearing **100%** for training, signifying the model's ability to detect nearly all objects.
- **Confidence** values, representing the likelihood of a predicted bounding box containing an object, showed a consistent upward trend, approaching **0.9** by the final epochs.

**Loss Components**:
- *Loss_classifier*: Cross-Entropy Loss for region classification.
- *Loss_box_reg*: Smooth L1 Loss for refining bounding box coordinates.
- *Loss_objectness*: Binary Cross-Entropy Loss for object/background classification.
- *Loss_rpn_box_reg*: Smooth L1 Loss for improving RPN proposals.

Despite these advancements, the confusion matrix revealed occasional misclassifications, particularly for overlapping classes. These instances primarily occurred due to tight IoU thresholds, which sometimes categorized valid predictions as "background."

**Precision and Recall Trends:**

![Precision Recall](https://prod-files-secure.s3.us-west-2.amazonaws.com/fa1b5077-0ed9-44f4-b1ae-33f8b9441bfb/cbf271db-a185-44a0-bfa1-c837c00b37d1/Screenshot_2025-01-07_alle_04.51.23.png)

**Confusion Matrix:**

![Confusion Matrix](https://prod-files-secure.s3.us-west-2.amazonaws.com/fa1b5077-0ed9-44f4-b1ae-33f8b9441bfb/fd3dd3f1-71d4-4f4b-8c16-63452699dcf7/Screenshot_2025-01-07_alle_21.31.46.png)

Faster R-CNN proved highly effective for object detection in satellite imagery, outperforming simpler models in precision and recall. The integration of RPN and FPN marked a significant step forward in handling complex datasets like xView.

---

Ecco la sezione aggiornata del README, discorsiva e dettagliata, basata sulle informazioni fornite:

---

## Detection Without Proposals: SSD and YOLO

Two modern approaches to object detection, **SSD (Single Shot Multibox Detector)** and **YOLO (You Only Look Once)**, aim to simplify the detection process by bypassing the region proposal stage seen in Faster R-CNN. Instead, these models directly predict bounding boxes and class labels from the input image in a single pass, making them faster and more efficient.

The underlying principle of both methods is to process the image through a convolutional neural network (CNN), dividing it into a grid. For each grid cell, the model predicts a set of bounding boxes along with confidence scores and class probabilities. This approach eliminates the need for explicit region proposals, resulting in significant improvements in inference speed while maintaining competitive accuracy.

### SSD: Single Shot Multibox Detector

For this project, we utilized the **SSD300 architecture** with a **VGG-16 backbone**, pre-trained on ImageNet. The fixed input size of 300x300 pixels ensures consistency and computational efficiency. The training process leveraged a pre-trained model fine-tuned on the VOC dataset, with adaptations to accommodate the 11 target classes (plus a background class) of our custom dataset.

#### Training Details
The SSD model was trained using:
- **Optimizer**: Stochastic Gradient Descent (SGD)  
- **Learning Rate**: Initial value of \(1 \times 10^{-3}\), adjusted using a learning rate scheduler  
- **Gradient Clipping**: To address exploding gradients  
- **Loss Function**: 
  - **SmoothL1Loss** for bounding box localization, measuring how closely predicted boxes match the ground truth.  
  - **CrossEntropy** for classification, assessing the model's ability to assign correct class probabilities.

The training focused on balancing speed and accuracy. While SSD generally sacrifices some precision compared to Faster R-CNN, its real-time processing capability makes it a practical choice for many applications.

#### Results
The SSD model showed notable improvements in bounding box localization and class predictions over the course of training. However, due to its focus on speed, precision was slightly lower compared to other methods. This trade-off highlights SSD's suitability for applications requiring faster inference, even at the cost of minor accuracy reductions.

---

### YOLO: You Only Look Once

**YOLO** adopts a similar single-shot detection approach but emphasizes global image understanding by processing the entire input image as a unified context. The model divides the image into a grid (e.g., 13x13 or 19x19) and predicts multiple bounding boxes for each grid cell, including their class probabilities and confidence scores. This allows YOLO to make rapid and robust predictions, particularly for large-scale datasets.

#### Implementation
Using the **Ultralytics YOLO** framework, we fine-tuned a pre-trained YOLOv11 model (`yolo11n.pt`) on our custom dataset. The configuration maintained a consistent image size of **640x640 pixels**—a standard in YOLO's pre-trained models to optimize compatibility and performance. To preserve learned features, the backbone's first three layers were frozen during training.

#### Training Experiments
Two optimization strategies were tested:
1. **Adam Optimizer**: With a learning rate of \(1 \times 10^{-3}\), the model was trained for 30 epochs.  
2. **SGD Optimizer**: With an initial learning rate of \(1 \times 10^{-2}\), decayed progressively to \(1 \times 10^{-3}\), also over 30 epochs.

The **SGD-trained model** outperformed its Adam counterpart across all metrics:
- **Box Precision (P)**: 0.604 (SGD) vs. 0.566 (Adam)  
- **Recall (R)**: 0.497 (SGD) vs. 0.485 (Adam)  
- **mAP50**: 0.514 (SGD) vs. 0.481 (Adam)  
- **mAP50-95**: 0.273 (SGD) vs. 0.252 (Adam)  

Given its superior performance, the SGD configuration was selected for the final 100-epoch training cycle.

---

### Comparative Analysis: SSD vs. YOLO

While both models excel in speed, YOLO's global image processing offers advantages in certain scenarios, such as minimizing false positives in complex contexts. SSD, on the other hand, benefits from its structured anchor-based approach, which can provide more reliable localization for simpler scenes.

#### Key Insights
- YOLO demonstrated a robust ability to generalize across classes, with fewer false positives in cluttered environments.  
- SSD maintained competitive accuracy but occasionally struggled with smaller or overlapping objects.  
- Both models faced challenges with underrepresented classes, underscoring the importance of dataset balancing and augmentation.

#### Example Predictions and Observations
- **YOLO Strengths**: YOLO excelled in distinguishing visually similar classes by leveraging its global context awareness. For example, it avoided common false positives, such as confusing road markings for vehicles.  
- **SSD Limitations**: Despite high recall, SSD occasionally produced redundant bounding boxes, highlighting the need for enhanced non-maximum suppression (NMS) strategies.

Overall, YOLO emerged as a better-suited model for our dataset, striking a balance between speed and accuracy, particularly after fine-tuning.

---

### SkyFusion Dataset

After testing our models on the xView dataset, we decided to validate their performance on a different dataset: **SkyFusion**. This dataset is specifically designed for detecting very small objects in satellite images, focusing on three primary classes: **Aircraft**, **Ship**, and **Vehicle**. 

SkyFusion consists of **2,996 images** with approximately **43,000 labeled objects**. As shown in the histogram, the dataset is **heavily imbalanced**, with certain classes being far more represented than others. Unlike xView, where we implemented balancing techniques, this experiment was intended as a straightforward **porting test** for the model, so no dataset balancing policies were applied. This allowed us to evaluate how the model performs in such challenging conditions.

---

### Fine-Tuning Faster R-CNN

To adapt Faster R-CNN to the SkyFusion dataset, we performed a **fine-tuning process** using the model previously pre-trained on the xView dataset. Specifically, we initialized training with the checkpoint saved at the **eighth epoch** from the xView training process. This choice leveraged the model's prior knowledge of satellite imagery features, while allowing it to specialize in the new dataset.

The fine-tuning process used the same hyperparameters as the initial Faster R-CNN training:
- Optimizer: **SGD**
- Learning Rate: **5e-4**
- Weight Decay: **5e-4**
- Momentum: **0.9**
- Training Duration: **10 epochs**

This approach ensured consistency and allowed for a meaningful comparison of the model's performance across the two datasets.

---

### Predictions on SkyFusion

Using the final model saved after the 10th epoch, we made predictions on the SkyFusion test set. Visual inspection revealed that the model performed reasonably well, particularly on large-scale images containing very small objects. The bounding boxes generated appeared accurate in many cases, successfully identifying Aircraft, Ships, and Vehicles in their respective contexts.

However, while these qualitative results seemed promising, a deeper analysis through the **confusion matrix** highlighted certain shortcomings.

---

### Confusion Matrix Analysis

The confusion matrix clearly illustrated the **challenges posed by the SkyFusion dataset**. Compared to xView, where the chunking process effectively increased object sizes and made detection easier, the SkyFusion dataset retained its original image scale. Consequently, objects were often smaller and more challenging to detect.

The model struggled with:
- **Missed detections**: A significant number of objects went undetected, likely due to their small size or similarity to the background.
- **Class confusion**: The imbalance in the dataset contributed to misclassifications, particularly for the less-represented classes.

Despite these difficulties, the model demonstrated its capacity to generalize to new datasets, even under less-than-ideal conditions. 

---

### Conclusion and Future Work

This final experiment marked the conclusion of our project. While there is certainly room for improvement—particularly in addressing dataset imbalances and refining detection of small objects—we are proud of the progress made. Testing multiple models and methodologies on diverse datasets has provided us with a deeper understanding of object detection techniques and their practical applications.

For the future, enhancements such as better handling of class imbalances, improved preprocessing for small object detection, and exploration of additional architectures like YOLOv8 could further elevate performance. Nevertheless, this project has been a valuable opportunity to explore the potential and limitations of state-of-the-art object detection models.
