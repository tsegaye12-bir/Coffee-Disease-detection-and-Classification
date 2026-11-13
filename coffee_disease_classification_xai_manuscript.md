# Hybrid Coffee Leaf Disease Classification with Explainable AI: A Pixel-Deduplicated JMuBEN Experiment

## Abstract

Automated coffee-leaf disease classification has reported very high performance on several image collections, but results can be misleading when duplicate files, class imbalance, and inconsistent test partitions are not audited. This study evaluates four pretrained convolutional classifiers on the user-provided JMuBEN archive: EfficientNet-B0, MobileNetV3-Large, ResNet-50, and a feature-fusion hybrid that concatenates EfficientNet-B0 and MobileNetV3-Large representations. The archive contains 58,549 readable JPEG files in five classes, but exact hashing of decoded RGB pixels identifies only 1,357 unique images; 57,192 repeated instances are removed before splitting. The retained counts are Cerscospora 82, Healthy 26, Leaf rust 521, Miner 547, and Phoma 181. A stratified 70:15:15 split yields 949 training, 204 validation, and 204 test images. Each trained model ran for 30 epochs using ImageNet initialization, online train-only augmentation, weighted cross-entropy, and AdamW. On this single held-out split, the hybrid obtained 99.02% accuracy and 98.64% macro-F1; MobileNetV3-Large and ResNet-50 each obtained 98.04% accuracy and approximately 96.91% macro-F1. ResNet-50 was selected by validation macro-F1 (98.54%), while the hybrid had the highest observed test score. Grad-CAM, Integrated Gradients, and occlusion were compared on 24 test images for both models with a top-10%-pixel deletion sensitivity check. Integrated Gradients produced the largest mean confidence drop (0.314 for ResNet-50; 0.301 for the hybrid), but the check is exploratory and not a causal faithfulness proof. The unusually high duplicate rate and only four Healthy examples in the test split sharply limit generalization claims. Results should be treated as a leakage-aware pilot benchmark, not evidence of field readiness or superiority over prior studies [4], [9], [15].

**Index Terms—** coffee leaf disease, deep learning, hybrid feature fusion, transfer learning, explainable artificial intelligence, JMuBEN, duplicate audit, Grad-CAM, Integrated Gradients.

## Introduction

**Background—** Coffee leaves show symptoms from distinct biotic stresses that matter for crop monitoring and classification. Coffee leaf rust, associated with *Hemileia vastatrix*, may produce chlorotic areas and orange powdery lesions, while Cercospora, Phoma, and leaf-miner damage present different visual cues [15]. Because symptoms may be visible on leaf surfaces, camera images have become a practical input to automated screening. Such systems do not replace field diagnosis: image appearance varies with disease stage, leaf age, lighting, camera, background, and which leaf surface is photographed. These sources of variation are part of the recognition problem, not incidental details.

The coffee-image literature has progressed from conventional image processing and classifiers to convolutional neural networks (CNNs), transfer learning, and multi-model systems. Montalbo and Hernandez compared deep CNN architectures for Barako coffee-leaf diseases [1]. Yebasse *et al.* examined disease visualization and classification on coffee images [2]. Novtahaning *et al.* evaluated an ensemble of fine-tuned CNNs for four disease categories [3]. Ayikpa *et al.* compared machine-learning and deep-learning approaches for coffee-leaf classification [4]. Work on Rwandan Arabica has tested deep-learning recognition in a geographically distinct setting [5]. Pham *et al.* compared early- and late-fusion strategies for coffee disease classification [6]. Other studies have considered edge deployment [7], image segmentation and feature concatenation [8], transfer-learning architecture selection [9], coffee-rust classification [10], interpretable CNN architectures [11], and real-time identification [12]. This body of work shows that coffee-leaf image classification is an active research area, while also making clear that reported performance depends on the dataset, class definitions, image partition, and evaluation protocol used in each study.

The datasets differ materially. Jepkoech *et al.* introduced the JMuBEN and JMuBEN2 Arabica coffee-leaf collections from Mutira plantation in Kirinyaga County, Kenya, with images acquired under field conditions and labels for five categories: Cerscospora, rust, Phoma, miner, and healthy [15]. Their paper describes 58,555 image files. Other studies use Barako coffee images [1], Robusta images such as RoCoLe [2], Rwandan Arabica leaves [5], or BRACOL material [6]. Some studies address a narrow task, such as rust versus healthy, while others classify multiple diseases, estimate stress severity, or evaluate deployment speed [2], [7], [10], [12], [16]. These differences mean that accuracy values across papers are not direct estimates of performance on the same task. A model tested on images from one plantation, device, or disease distribution may not maintain its performance on another. Reviews of plant-disease recognition likewise emphasize dataset design, image variability, and evaluation protocol as central considerations when interpreting deep-learning results [18].

**Existing gap—** The use of ensembles and hybrid representations is not new in coffee classification. Novtahaning *et al.* combined selected pretrained networks [3]. Pham *et al.* studied feature fusion [6]. Abuhayi and Mossa concatenated features from GoogLeNet and ResNet-based networks [8]. Faisal *et al.* explicitly examined hybrid feature fusion for coffee-leaf disease classification [25]. Explainability is also being explored: Yebasse *et al.* included visualization in coffee-disease analysis [2], and Chavarro *et al.* compared multiple class-activation maps on coffee-rust images with perturbation-based evaluation [11]. Therefore, the research gap addressed here is not the absence of hybrid models or XAI in the broader literature.

Instead, three narrower issues motivate this study. First, the published JMuBEN/JMuBEN2 collection comprises tens of thousands of files, and Mansouri *et al.* report transfer-learning results on a collection of 58,555 images [9]. The archive supplied for the present experiment contains 58,549 readable files, a difference of six from the published count whose cause is unknown. Second, the file count does not measure independent visual examples: repeated files may be byte-different encodings of the same decoded image. A large nominal dataset can consequently represent far fewer distinct images than its folder count suggests. Third, papers may report validation, test, or combined held-out metrics under different split designs. These reporting differences complicate comparison, particularly when the apparent scores are very high. Coffee-disease research also shows that architecture, preprocessing, and hyperparameter choices affect measured performance [20]. Robust evaluation therefore requires auditing the supplied data, separating model selection from held-out testing, and reporting class-level supports alongside aggregate metrics.

The need for interpretable outputs is related but distinct. A classifier can be accurate on a test partition while using image regions that are irrelevant to pathology, such as background, shadows, or capture artifacts. Visual attribution methods can reveal which pixels or regions influenced a score, but a heatmap is not itself proof that the model learned disease biology. Chavarro *et al.* compared seven class-activation methods and used ROAD-based perturbation evaluation for coffee-rust classification [11]. General XAI literature similarly cautions that explanations require task-appropriate evaluation and should not be accepted solely because a visualization appears plausible [19]. The present experiment therefore compares Grad-CAM, Integrated Gradients, and occlusion, and adds a narrowly defined pixel-deletion sensitivity check; it does not treat the resulting maps as lesion masks or as validated agronomic explanations.

**Problem statement—** The practical problem is to determine what can be concluded from training and testing on this particular user-supplied JMuBEN archive when many file entries may encode repeated pixels and the remaining unique classes may be imbalanced. If identical decoded images are placed in both training and test partitions, a model may recognize a repeated image rather than generalize to new leaves. For that reason, counting 58,549 files as independent observations would overstate the effective sample size. In this archive, exact decoded-RGB deduplication leaves only 1,357 distinct images, including 26 Healthy images. The small retained sample also makes a single test score uncertain, especially for classes with few independent examples. The methodological question is therefore not simply which architecture gives the largest accuracy value. It is whether a feature-fusion hybrid offers a useful result against individual pretrained CNNs when all models use the same deduplicated split, and how class-specific errors and explanations qualify that result.

**Objectives—** The study has four objectives. First, it audits the uploaded archive for image readability, class counts, and exact duplicate decoded RGB content, then documents the resulting unique-image sample. Second, it trains three pretrained CNN baselines—EfficientNet-B0, MobileNetV3-Large, and ResNet-50—and a predeclared EfficientNet–MobileNet feature-fusion model for 30 epochs under one shared split and training protocol. Third, it compares the models using held-out accuracy, macro- and weighted-averaged precision, recall and F1, full per-class reports, confusion matrices, and 30-epoch accuracy/loss curves. Fourth, it examines Grad-CAM, Integrated Gradients, and occlusion for the validation-selected ResNet-50 and the predeclared hybrid on matched test images, using top-10%-pixel deletion only as an exploratory sensitivity measure.

The contributions are: **(i)** an exact decoded-pixel audit of the uploaded archive that documents the reduction from 58,549 files to 1,357 unique images and establishes a duplicate-aware split for this experiment; **(ii)** a controlled comparison of three established pretrained CNNs with an EfficientNet–MobileNet feature-fusion hybrid, where all models use the same train/validation/test partitions and training conditions; **(iii)** class-level test reports, confusion matrices, and learning curves presented together for each model so the aggregate score can be interpreted against its specific errors and training trajectory; and **(iv)** matched XAI visualizations and deletion-sensitivity summaries for the validation-selected baseline and the predeclared hybrid, with the limitations of that check stated explicitly. These contributions are specific to this archive and experiment. They are not claims to introduce hybrid coffee classification, feature fusion, or coffee-image explainability for the first time [3], [6], [8], [11], [25].

## Materials and Methods

### Dataset and audit

The input was the user-uploaded `12.zip`, which expands to the `JMuBEN` class-folder structure. The five source folder names were preserved: Cerscospora, Healthy, Leaf rust, Miner, and Phoma. The archive contained 58,549 JPEG files: Cerscospora 7,681; Healthy 18,983; Leaf rust 8,336; Miner 16,978; and Phoma 6,571. All 58,549 files opened successfully; no corrupt images or decoded-pixel conflicts across different class labels were found. The dominant raw image format was RGB JPEG at 128 × 128 pixels (55,596 files); the remaining files had varied dimensions. Images were resized to 160 × 160 for model input.

The published JMuBEN/JMuBEN2 total cited by Mansouri *et al.* is 58,555 [9], six higher than this archive’s 58,549 files. The reason for the difference was not established. More importantly, byte-level hashing found repeated files with different JPEG encodings that decoded to identical images. The audit therefore converted every file to RGB, prepended its decoded width and height, and SHA-256-hashed the resulting pixel bytes. One representative from each same-label decoded-pixel group was retained. This removed 57,192 duplicate instances and yielded 1,357 unique decoded RGB images, with counts in Table I. This is exact decoded-pixel deduplication, not a complete perceptual or source-provenance duplicate analysis.

**Table I. Raw archive files and retained unique decoded RGB images**

| Class | Raw files | Unique decoded RGB images | Duplicate instances removed |
|---|---:|---:|---:|
| Cerscospora | 7,681 | 82 | 7,599 |
| Healthy | 18,983 | 26 | 18,957 |
| Leaf rust | 8,336 | 521 | 7,815 |
| Miner | 16,978 | 547 | 16,431 |
| Phoma | 6,571 | 181 | 6,390 |
| **Total** | **58,549** | **1,357** | **57,192** |

### Partitioning, augmentation, and sample accounting

After decoded-pixel deduplication, a seeded stratified split (seed 42) allocated 70% to training, 15% to validation, and 15% to testing. The split counts are shown in Table II. The test set was not used for checkpoint or architecture selection. The validation macro-F1 selected ResNet-50; the hybrid was also explained because feature fusion was the predeclared focus, not because of its test result.

**Table II. Unique-image split by class**

| Class | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Cerscospora | 57 | 12 | 13 | 82 |
| Healthy | 18 | 4 | 4 | 26 |
| Leaf rust | 364 | 79 | 78 | 521 |
| Miner | 383 | 82 | 82 | 547 |
| Phoma | 127 | 27 | 27 | 181 |
| **Total** | **949** | **204** | **204** | **1,357** |

Augmentation was online and applied only to training images: random resized crop (scale 0.82–1.00; aspect ratio 0.90–1.10), horizontal flip (0.5 probability), vertical flip (0.25 probability), random rotation within ±15°, and mild brightness/contrast/saturation jitter. Validation and test images received deterministic resizing and ImageNet normalization only. No offline augmented images were materialized. Thus, the dataset remained 1,357 unique source images; each epoch presented 949 training images with a newly sampled training transform, equivalent to 28,470 image-presentations per trained model over 30 epochs. This should not be mistaken for 28,470 unique examples.

### Models and hybrid feature fusion

Four classifiers were trained. **EfficientNet-B0**, **MobileNetV3-Large**, and **ResNet-50** used their TorchVision ImageNet pretrained weights and a five-class output layer. For compute-efficient fine-tuning, the final feature stage plus classifier was trainable; earlier layers were frozen. The **HybridEfficientMobile** model used both pretrained backbones, removed their classification heads, concatenated the 1,280-dimensional EfficientNet-B0 pooled feature vector and 960-dimensional MobileNetV3-Large pooled feature vector, then passed the 2,240-dimensional vector to a fusion MLP: 512 units, batch normalization, SiLU, 0.35 dropout; 256 units, batch normalization, SiLU, 0.25 dropout; and a five-class output layer. The final feature stage of each branch and the fusion head were trainable, with the earlier layers frozen. A probability-averaged soft-voting ensemble of the three single-backbone models was computed after fitting; it did not receive a separate training run.

### Training hyperparameters and evaluation

All trained models ran for exactly 30 epochs on CPU (six threads; no CUDA GPU). Input size was 160 × 160 RGB, normalized with ImageNet mean and standard deviation. Batch size was 24. AdamW used a backbone learning rate of 1 × 10⁻⁵, a classifier/fusion-head learning rate of 1 × 10⁻⁴, and weight decay 1 × 10⁻⁴. The loss was class-weighted cross-entropy with label smoothing 0.05; class weights were computed from the training partition using inverse-square-root class frequency and normalized to mean one. A cosine annealing schedule ran over 30 epochs. Seed 42 controlled the split and randomized transforms. For each model, the checkpoint with the highest validation macro-F1 was retained. The locked test set was then evaluated once.

Reported measures are accuracy; macro-precision, macro-recall, and macro-F1; weighted precision, recall, and F1; and class-specific precision, recall, F1, and support. Macro metrics give each class equal influence, while weighted metrics reflect test prevalence. Confusion matrices and training/validation accuracy and loss curves are supplied with this manuscript. Given the small and imbalanced unique-image sample, results are descriptive for this split and do not estimate performance on independent farms or locations.

### XAI methods and deletion sensitivity

Grad-CAM was computed from a final convolutional block, Integrated Gradients used a zero-valued normalized baseline and 24 integration steps, and Captum Occlusion used 32 × 32 patches with 24-pixel spatial strides. The same 24 test examples were explained for ResNet-50 and the hybrid: five per class where available, and all four Healthy test images. The model attribution target was its predicted class. For a simple sensitivity comparison, the top 10% of pixels in each normalized attribution map were replaced with the normalized zero baseline (approximately the ImageNet mean pixel), and the predicted-class probability drop was recorded. The reported mean and standard deviation are across 24 images per model and method. This single-step deletion check is exploratory: it is sensitive to baseline and masking choices and is not a causal explanation metric or proof of faithfulness [19].

## Results and Analysis

All four trained classifiers completed 30 epochs using the same unique-image split. The test set contains Cerscospora 13, Healthy 4, Leaf rust 78, Miner 82, and Phoma 27 images (204 total). Table III provides the compact cross-model comparison; the following A–D subsections give the architecture-specific result, full class report, confusion matrix, and training/validation curves together. ResNet-50 was selected by the prespecified validation macro-F1 rule. The hybrid is also reported because feature fusion was the predeclared study focus; it was not selected using test results.

**Table III. Validation selection and held-out test performance**

| Model | Best epoch | Validation macro-F1 | Test accuracy | Macro precision | Macro recall | Test macro-F1 | Weighted F1 | Training time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 23 | 0.9015 | 0.9363 | 0.8763 | 0.9626 | 0.9132 | 0.9376 | 4.48 |
| MobileNetV3-Large | 15 | 0.9588 | 0.9804 | 0.9666 | 0.9722 | 0.9691 | 0.9806 | 2.81 |
| ResNet-50 | 29 | **0.9854** | 0.9804 | 0.9664 | 0.9722 | 0.9690 | 0.9806 | 10.35 |
| EfficientNet–MobileNet feature-fusion hybrid | 22 | 0.9758 | **0.9902** | **0.9786** | **0.9950** | **0.9864** | **0.9903** | 5.94 |
| Soft-voting ensemble (three single models) | — | — | 0.9804 | 0.9666 | 0.9722 | 0.9691 | 0.9806 | — |

### A. EfficientNet-B0

**Architecture and performance.** ImageNet-pretrained EfficientNet-B0 was fine-tuned with its final feature stage and five-class head trainable. Its validation macro-F1 checkpoint occurred at epoch 23 (0.9015). On the locked test set it reached 93.63% accuracy, 0.8763 macro-precision, 0.9626 macro-recall, 0.9132 macro-F1, and 0.9376 weighted F1; training required 4.48 minutes. The comparatively lower macro-precision reflects more false-positive predictions across several low-support classes.

**Per-class classification report (Table IV-A).** EfficientNet-B0 classified 13/13 Cerscospora, 4/4 Healthy, 72/78 Leaf rust, 76/82 Miner, and 26/27 Phoma examples correctly. The Healthy recall is based on only four test images.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cerscospora | 0.765 | 1.000 | 0.867 | 13 |
| Healthy | 0.800 | 1.000 | 0.889 | 4 |
| Leaf rust | 1.000 | 0.923 | 0.960 | 78 |
| Miner | 0.950 | 0.927 | 0.938 | 82 |
| Phoma | 0.867 | 0.963 | 0.912 | 27 |

**Confusion matrix (Figure 1).** The matrix uses actual classes as rows and predicted classes as columns; most errors are concentrated in Leaf rust and Miner, consistent with their lower class recalls.

![EfficientNet-B0 test confusion matrix](/home/ubuntu/coffee_xai/results_jmuben/EfficientNetB0_confusion_matrix.png)

**Accuracy and loss curves (Figure 2).** Training and validation curves over all 30 epochs show the optimization trajectory; the epoch-23 checkpoint was selected by validation macro-F1.

![EfficientNet-B0 accuracy and loss curves](/home/ubuntu/coffee_xai/results_jmuben/EfficientNetB0_curves.png)

### B. MobileNetV3-Large

**Architecture and performance.** ImageNet-pretrained MobileNetV3-Large was fine-tuned with the final feature stage and five-class head trainable. Its best validation macro-F1 was 0.9588 at epoch 15. Test accuracy was 98.04%, macro-precision 0.9666, macro-recall 0.9722, macro-F1 0.9691, and weighted F1 0.9806; training took 2.81 minutes. It tied ResNet-50 in test accuracy and slightly exceeded it in macro-F1 at the reported precision.

**Per-class classification report (Table IV-B).** Correct counts were 12/13 Cerscospora, 4/4 Healthy, 77/78 Leaf rust, 81/82 Miner, and 26/27 Phoma.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cerscospora | 0.857 | 0.923 | 0.889 | 13 |
| Healthy | 1.000 | 1.000 | 1.000 | 4 |
| Leaf rust | 1.000 | 0.987 | 0.994 | 78 |
| Miner | 0.976 | 0.988 | 0.982 | 82 |
| Phoma | 1.000 | 0.963 | 0.981 | 27 |

**Confusion matrix (Figure 3).** The matrix shows one missed Cerscospora, one Leaf rust, one Miner, and one Phoma example; all four Healthy cases were correct.

![MobileNetV3-Large test confusion matrix](/home/ubuntu/coffee_xai/results_jmuben/MobileNetV3Large_confusion_matrix.png)

**Accuracy and loss curves (Figure 4).** The 30-epoch curves show rapid convergence followed by a relatively stable validation trajectory; the saved model is the epoch-15 validation macro-F1 checkpoint.

![MobileNetV3-Large accuracy and loss curves](/home/ubuntu/coffee_xai/results_jmuben/MobileNetV3Large_curves.png)

### C. ResNet-50

**Architecture and performance.** ImageNet-pretrained ResNet-50 used its final residual stage and five-class head for fine-tuning. It achieved the highest validation macro-F1, 0.9854, at epoch 29, and therefore was selected by the prespecified validation-only rule. Test accuracy was 98.04%, macro-precision 0.9664, macro-recall 0.9722, macro-F1 0.9690, and weighted F1 0.9806; training took 10.35 minutes, the longest of the four models.

**Per-class classification report (Table IV-C).** Correct counts were 12/13 Cerscospora, 4/4 Healthy, 77/78 Leaf rust, 81/82 Miner, and 26/27 Phoma.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cerscospora | 0.857 | 0.923 | 0.889 | 13 |
| Healthy | 1.000 | 1.000 | 1.000 | 4 |
| Leaf rust | 0.987 | 0.987 | 0.987 | 78 |
| Miner | 0.988 | 0.988 | 0.988 | 82 |
| Phoma | 1.000 | 0.963 | 0.981 | 27 |

**Confusion matrix (Figure 5).** The matrix contains four test errors: one each in Cerscospora, Leaf rust, Miner, and Phoma; the four Healthy examples are correctly classified.

![ResNet-50 test confusion matrix](/home/ubuntu/coffee_xai/results_jmuben/ResNet50_confusion_matrix.png)

**Accuracy and loss curves (Figure 6).** The plotted train/validation accuracy and loss trajectories cover all 30 epochs. The late best epoch is consistent with continued validation improvement under this single run, but does not establish that 30 epochs is optimal across seeds.

![ResNet-50 accuracy and loss curves](/home/ubuntu/coffee_xai/results_jmuben/ResNet50_curves.png)

### D. EfficientNet–MobileNet feature-fusion hybrid

**Architecture and performance.** The hybrid concatenated the 1,280-dimensional EfficientNet-B0 and 960-dimensional MobileNetV3-Large pooled features before the fusion MLP. Its validation macro-F1 was 0.9758 at epoch 22. On the held-out test set it produced the highest observed point estimates: 99.02% accuracy, 0.9786 macro-precision, 0.9950 macro-recall, 0.9864 macro-F1, and 0.9903 weighted F1. Training took 5.94 minutes. It correctly classified 202 of 204 test images, including all four Healthy and all 27 Phoma images. This result is descriptive for one test split and is not a statistical significance claim.

**Per-class classification report (Table IV-D).** Correct counts were 13/13 Cerscospora, 4/4 Healthy, 77/78 Leaf rust, 81/82 Miner, and 27/27 Phoma.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cerscospora | 0.929 | 1.000 | 0.963 | 13 |
| Healthy | 1.000 | 1.000 | 1.000 | 4 |
| Leaf rust | 1.000 | 0.987 | 0.994 | 78 |
| Miner | 1.000 | 0.988 | 0.994 | 82 |
| Phoma | 0.964 | 1.000 | 0.982 | 27 |

**Confusion matrix (Figure 7).** The two errors are a Leaf rust image predicted as Cerscospora and a Miner image predicted as Phoma. The matrix also makes clear that the strong Healthy result is based on only four test examples.

![EfficientNet–MobileNet hybrid test confusion matrix](/home/ubuntu/coffee_xai/results_jmuben/HybridEfficientMobile_confusion_matrix.png)

**Accuracy and loss curves (Figure 8).** The 30-epoch curves show training and validation performance; epoch 22 was retained using validation macro-F1. The hybrid’s test-leading result should be interpreted together with its lower validation macro-F1 than ResNet-50, not as evidence that the test set determined model selection.

![Hybrid accuracy and loss curves](/home/ubuntu/coffee_xai/results_jmuben/HybridEfficientMobile_curves.png)

### E. Soft-voting ensemble

The probability-level soft-voting ensemble of EfficientNet-B0, MobileNetV3-Large, and ResNet-50 achieved 98.04% accuracy, 0.9666 macro-precision, 0.9722 macro-recall, 0.9691 macro-F1, and 0.9806 weighted F1. Its aggregate scores matched MobileNetV3-Large on this test set and did not exceed the feature-fusion hybrid. The ensemble is an arithmetic combination of trained model probabilities, not a separately trained architecture.

### F. Explainability comparison

The validation-selected ResNet-50 and predeclared fusion hybrid were each explained on the same 24 test images. Integrated Gradients yielded the largest average confidence drop after deletion of its top 10% pixels for both models: 0.314 ± 0.280 for ResNet-50 and 0.301 ± 0.304 for the hybrid. Grad-CAM yielded 0.253 ± 0.222 and 0.174 ± 0.186, respectively. Occlusion yielded 0.017 ± 0.177 for ResNet-50 and 0.144 ± 0.196 for the hybrid. The wide standard deviations indicate substantial image-to-image variation; the small ResNet-50 occlusion mean suggests that its most salient occlusion patches did not consistently align with the same single-pixel deletion test. These values compare sensitivity under one mask policy only and should not be interpreted as a definitive ranking of explanation faithfulness.

**Table V. Top-10%-pixel deletion sensitivity on 24 matched test images per model (mean ± SD confidence drop)**

| Explained model | Grad-CAM | Integrated Gradients | Occlusion |
|---|---:|---:|---:|
| ResNet-50 (validation-selected) | 0.253 ± 0.222 | **0.314 ± 0.280** | 0.017 ± 0.177 |
| EfficientNet–MobileNet hybrid (predeclared) | 0.174 ± 0.186 | **0.301 ± 0.304** | 0.144 ± 0.196 |

**Figure 9.** Mean top-10%-pixel deletion sensitivity by model and attribution method; error bars show one standard deviation across 24 matched test images.

![XAI deletion-sensitivity comparison](/home/ubuntu/coffee_xai/results_jmuben/xai_deletion_comparison.png)

**Figure 10.** Three attribution methods on the same held-out Leaf rust image for the validation-selected ResNet-50 and the hybrid. The occlusion map is patch-based and therefore appears coarser; these overlays are illustrative, not lesion segmentations.

![Matched-image XAI examples](/home/ubuntu/coffee_xai/results_jmuben/xai_example_grid.png)

## Discussion

### Comparison with recent studies

Table VI compares seven recent coffee-disease studies with the present experiment. Their metrics are not a leaderboard: task definitions, source image counts, acquisition conditions, split design, validation/test terminology, and independent-image counts differ. The three closest references are the JMuBEN dataset paper [15], Ayikpa *et al.* [4], and Mansouri *et al.* [9]. The latter reports results on the JMuBEN/JMuBEN2 collection and very high EfficientNet-B0 and ResNet-50 accuracy, but describes the non-training partition as combined validation/test, so its test estimate is not directly comparable to this fixed 15% test partition [9].

**Table VI. Seven recent studies and the present JMuBEN experiment**

| Study | Dataset/task and method | Reported result | Comparability caveat |
|---|---|---|---|
| Montalbo and Hernandez, 2020 [1] | 4,667 Barako images; four classes; VGG16, Xception, ResNetV2-152 | Validation accuracy 97%, 95%, 91% | 4,023/644 training-validation partition; no external test in the source abstract. |
| Yebasse *et al.*, 2021 [2] | RoCoLe, 1,560 Robusta images; binary rust/healthy; CNN patch guidance | Naïve test accuracy 77%; guided model reported at 98% validation accuracy | Inconsistent test/validation wording and patch-based dataset expansion; XAI qualitative. |
| Novtahaning *et al.*, 2022 [3] | 1,300 images; five classes; ensemble of EfficientNet-B0, ResNet-152, and VGG-16 | Accuracy 97.3%; F1 95.1% | Holdout/validation terminology differs; confusion matrix contains 248 examples. |
| Ayikpa *et al.*, 2022 [4] | JMuBEN Arabica coffee leaves; classical ML and custom CNN | Abstract reports DL performance from 97.37% to 100% | Split and sample details require caution; the stated range is not directly comparable to this deduplicated split. |
| Mansouri *et al.*, 2024 [9] | JMuBEN/JMuBEN2, 58,555 files, five classes; EfficientNet-B0, ResNet-50, MobileNet variants | EfficientNet-B0 99.99%; ResNet-50 99.91%; MobileNetV1/V2 99.52%/99.62% | 80% training; remainder described as combined validation/test. Duplicate audit and class-wise unique test supports differ. |
| Topal *et al.*, 2024 [10] | 1,685 Arabica images; five-class biotic stress and severity tasks | Best stress accuracy 95.65%; best severity accuracy 90.11% | Different source and task labels; authors note class imbalance and background bias. |
| Adelaja and Pranggono, 2025 [12] | BRACOL coffee images; five classes; real-time CNN comparison | VGG16 test accuracy 99.89% | Source images and derived examples differ; numeric split counts and real-world conditions are not equivalent. |
| **This study** | Uploaded JMuBEN archive: 58,549 files → 1,357 decoded-pixel-unique images; five classes; three CNN baselines plus feature-fusion hybrid | Hybrid test accuracy 99.02%; macro-F1 98.64% | Single 70/15/15 split; only 26 unique Healthy examples overall and four in test; exploratory XAI deletion check. |

The hybrid’s observed test macro-F1 exceeded the best single backbones by about 1.74 percentage points, and its test accuracy was higher by 0.98 points. This is consistent with the premise that combining two complementary pretrained representations can improve class separation, but the experiment does not isolate fusion as the cause: there is one seed, one split, a small unique-image pool, and the validation-selected model was ResNet-50 rather than the hybrid. The hybrid’s validation macro-F1 (0.9758) was below ResNet-50’s (0.9854), while its test result was higher. That validation/test ordering reversal emphasizes split variability; it is not evidence that the hybrid is globally superior.

The dataset audit is the dominant result and interpretive constraint. A raw archive close in size to the 58,555 entries in the published JMuBEN/JMuBEN2 study [9] contained only 1,357 unique decoded RGB arrays. The raw file count was therefore a poor proxy for independent visual diversity. The most severe class was Healthy: 18,983 files collapsed to only 26 decoded-pixel-unique images. Weighted loss reduces the influence of class prevalence during optimization, but neither class weighting nor online augmentation creates new independent biological examples. The four Healthy test cases yield an observed recall of 1.0 for all four models, but one missed case would have changed that recall by 0.25; this estimate is intrinsically unstable.

All four classifiers achieved strong point estimates on this particular test split, but they did not behave identically. EfficientNet-B0 had the lowest aggregate performance and lower precision for Cerscospora, Healthy, and Phoma. MobileNetV3-Large and ResNet-50 tied in accuracy and nearly tied in macro-F1. The hybrid improved Cerscospora precision while preserving high recall across classes. These per-class differences, especially for the low-support classes, matter more than a small aggregate accuracy gap. Comparison to prior coffee papers must remain qualified because some report validation or training results as “accuracy,” use patches or augmented copies, or combine validation and test partitions [1]–[4], [9], [12]. Coffee studies also show that capture backgrounds and acquisition conditions can affect apparent performance [2], [10], [12], while hyperparameter choices can materially shift coffee-rust results [20].

For XAI, Integrated Gradients produced the largest mean confidence drop under the chosen pixel-deletion policy for both ResNet-50 and the hybrid. This does not establish that it is the most faithful or agronomically meaningful explanation. Its standard deviation was approximately as large as its mean, and deletion sensitivity depends on baseline, ranking resolution, and patch/pixel semantics. Occlusion uses 32 × 32 regions, while the deletion check removes pixels from the top-ranked map; this mismatch likely contributes to the weaker ResNet-50 occlusion score. The example maps show method-specific patterns but do not constitute lesion localization. A stronger comparison would use multiple deletion/insertion baselines, ROAD or other perturbation metrics, random-map controls, and expert-reviewed lesion masks [2], [11], [19].

### Limitations

The principal limitation is the archive itself: 57,192 of 58,549 file entries repeat decoded pixel content, and Healthy has only 26 unique images. Exact RGB pixel deduplication prevents exact copies from crossing the split, but the audit did not fully identify perceptual near-duplicates, the same physical leaf photographed differently, or related images from one plant/session. Thus the split may still contain source-level dependence. The 204-image test set includes only four Healthy examples and 13 Cerscospora examples, making class-specific point estimates uncertain.

The experiment uses one random seed and one image-level split, no external farm or geographic test set, and no independent expert adjudication of labels. Fine-tuning is partial, input resolution is 160 × 160, and only one hyperparameter recipe was evaluated. The XAI deletion check used a single baseline and a single 10% mask fraction; it is a sensitivity analysis, not a validated faithfulness benchmark. No field lighting, blur, phone-camera, or device robustness testing was conducted. The results therefore cannot support operational deployment or superiority claims over studies with different data partitions.

## Conclusion and Recommendations

This study completed 30-epoch training for EfficientNet-B0, MobileNetV3-Large, ResNet-50, and an EfficientNet–MobileNet feature-fusion hybrid on a user-uploaded five-class JMuBEN archive. The hybrid led the single held-out split with 99.02% accuracy and 98.64% macro-F1; ResNet-50 was selected by validation macro-F1. Integrated Gradients produced the largest mean confidence reduction in the exploratory top-10%-pixel deletion check, but high variance and the limited perturbation design prevent a strong faithfulness conclusion.

The most important finding is not the nominal 99% score: it is that 58,549 files contained only 1,357 unique decoded RGB images, including just 26 Healthy images. Accordingly, the scores must be described as preliminary and split-specific. Before any claim of improved disease screening, the archive should be checked with its provider, duplicate/source groups documented, and additional independent Healthy and Cerscospora images collected. Follow-up work should use repeated grouped or stratified cross-validation, larger source-diverse test sets, external farm images, robustness tests, and expert-reviewed lesion annotations. Hybrid fusion is promising within this pilot, but confirmation requires repeated seeds and independent evaluation. Model predictions and heatmaps are research outputs, not a substitute for expert agronomic diagnosis.

## References

[1] F. J. P. Montalbo and A. A. Hernandez, “Classifying Barako coffee leaf diseases using deep convolutional models,” *International Journal of Advances in Intelligent Informatics*, vol. 6, no. 2, pp. 197–209, 2020, doi: 10.26555/ijain.v6i2.495.

[2] M. Yebasse, B. Shimelis, H. Warku, J. Ko, and K. J. Cheoi, “Coffee disease visualization and classification,” *Plants*, vol. 10, no. 6, Art. no. 1257, 2021, doi: 10.3390/plants10061257.

[3] D. Novtahaning, H. A. Shah, and J.-M. Kang, “Deep learning ensemble-based automated and high-performing recognition of coffee leaf disease,” *Agriculture*, vol. 12, no. 11, Art. no. 1909, 2022, doi: 10.3390/agriculture12111909.

[4] K. J. Ayikpa, D. Mamadou, P. Gouton, and K. J. Adou, “Experimental evaluation of coffee leaf disease classification and recognition based on machine learning and deep learning algorithms,” *Journal of Computer Science*, vol. 18, no. 12, pp. 1201–1212, 2022, doi: 10.3844/jcssp.2022.1201.1212.

[5] E. Hitimana *et al*., “An intelligent system-based coffee plant leaf disease recognition using deep learning techniques on Rwandan Arabica dataset,” *Technologies*, vol. 11, no. 5, Art. no. 116, 2023, doi: 10.3390/technologies11050116.

[6] T. C. Pham, V. D. Nguyen, C. H. Le, M. Packianather, and V.-D. Hoang, “Artificial intelligence-based solutions for coffee leaf disease classification,” in *IOP Conference Series: Earth and Environmental Science*, vol. 1278, Art. no. 012004, 2023, doi: 10.1088/1755-1315/1278/1/012004.

[7] J. V. Y. B. Yamashita and J. P. R. R. Leite, “Coffee disease classification at the edge using deep learning,” *Smart Agricultural Technology*, vol. 4, Art. no. 100183, 2023, doi: 10.1016/j.atech.2023.100183.

[8] B. M. Abuhayi and A. A. Mossa, “Coffee disease classification using convolutional neural network based on feature concatenation,” *Informatics in Medicine Unlocked*, vol. 39, Art. no. 101245, 2023, doi: 10.1016/j.imu.2023.101245.

[9] N. Mansouri, H. Guesmi, and A. Alkhalil, “A deep learning model for detection and classification of coffee-leaf diseases using the transfer-learning technique,” *International Journal of Advances in Intelligent Informatics*, vol. 10, no. 3, pp. 379–396, 2024, doi: 10.26555/ijain.v10i3.1573.

[10] A. Topal, B. Tunga, and E. B. Tirkolaee, “DeepEMPR: Coffee leaf disease detection with deep learning and enhanced multivariance product representation,” *PeerJ Computer Science*, vol. 10, Art. no. e2406, 2024, doi: 10.7717/peerj-cs.2406.

[11] A. Chavarro, D. Renza, and E. Moya-Albor, “ConvNext as a basis for interpretability in coffee leaf rust classification,” *Mathematics*, vol. 12, no. 17, Art. no. 2668, 2024, doi: 10.3390/math12172668.

[12] O. Adelaja and B. Pranggono, “Leveraging deep learning for real-time coffee leaf disease identification,” *AgriEngineering*, vol. 7, no. 1, Art. no. 13, 2025, doi: 10.3390/agriengineering7010013.

[13] Tran Huu Quyet, “Coffee leaf disease classification using three lightweight deep learning models: Robustness evaluation against image degradation and Grad-CAM visualization,” *Scientific Journal of Tan Trao University*, vol. 12, no. 2, 2026, doi: 10.51453/3093-3706/2026/1478.

[14] C. C. R. Situmorang, “ResNet-34 architectural approach in convolutional neural networks (CNN) for classifying diseases in coffee leaves,” undergraduate thesis, Universitas Medan Area, Medan, Indonesia, 2026. [Online]. Available: https://repositori.uma.ac.id/handle/123456789/30262

[15] J. Jepkoech, D. M. Mugo, B. K. Kenduiywo, and E. C. Too, “Arabica coffee leaf images dataset for coffee leaf disease detection and classification,” *Data in Brief*, vol. 36, Art. no. 107142, 2021, doi: 10.1016/j.dib.2021.107142.

[16] J. G. M. Esgario, R. A. Krohling, and J. A. Ventura, “Deep learning for classification and severity estimation of coffee leaf biotic stress,” *Computers and Electronics in Agriculture*, vol. 169, Art. no. 105162, 2020, doi: 10.1016/j.compag.2019.105162.

[17] Y. Rodriguez-Gallo, B. Escobar-Benitez, and J. Rodriguez-Lainez, “Robust coffee rust detection using UAV-based aerial RGB imagery,” *AgriEngineering*, vol. 5, no. 3, pp. 1415–1431, 2023, doi: 10.3390/agriengineering5030088.

[18] M. Shoaib *et al*., “An advanced deep learning models-based plant disease detection: A review of recent research,” *Frontiers in Plant Science*, vol. 14, Art. no. 1158933, 2023, doi: 10.3389/fpls.2023.1158933.

[19] A. B. Arrieta *et al*., “Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI,” *Information Fusion*, vol. 58, pp. 82–115, 2020, doi: 10.1016/j.inffus.2019.12.012.

[20] A. F. Chavarro, D. Renza, and D. M. Ballesteros, “Influence of hyperparameters in deep learning models for coffee rust detection,” *Applied Sciences*, vol. 13, no. 7, Art. no. 4565, 2023, doi: 10.3390/app13074565.

[21] W. Albattah, M. Nawaz, A. Javed, M. Masood, and S. Albahli, “A novel deep learning method for detection and classification of plant diseases,” *Complex & Intelligent Systems*, vol. 8, no. 1, pp. 507–524, 2022, doi: 10.1007/s40747-021-00536-1.

[22] X. Liu, W. Min, S. Mei, L. Wang, and S. Jiang, “Plant disease recognition: A large-scale benchmark dataset and a visual region and loss reweighting approach,” *IEEE Transactions on Image Processing*, vol. 30, pp. 2003–2015, 2021, doi: 10.1109/TIP.2021.3049334.

[23] A. Abade, P. A. Ferreira, and F. de Barros Vidal, “Plant diseases recognition on images using convolutional neural networks: A systematic review,” *Computers and Electronics in Agriculture*, vol. 185, Art. no. 106125, 2021, doi: 10.1016/j.compag.2021.106125.

[24] B. Yoseph, “Ethiopian Coffee Leaf Disease,” *Kaggle* [Online]. Available: https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease. Accessed: Nov. 9, 2026. This separate collection is not the JMuBEN archive analyzed here.

[25] M. Faisal, J.-S. Leu, and J. T. Darmawan, “Model selection of hybrid feature fusion for coffee leaf disease classification,” *IEEE Access*, vol. 11, pp. 62281–62291, 2023, doi: 10.1109/ACCESS.2023.3286935.
