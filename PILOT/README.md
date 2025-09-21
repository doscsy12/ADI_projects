# Pilot projects

## Banknote Authentication
### Background
Despite the rise of digital payments, cash is still widely used in many Asian countries, especially in rural areas and informal economies. This is due to:
<br> (1) Limited access to banking or digital infrastructure.
<br> (2) Cash being faster and more trusted in low-tech or low-income communities.
<br> (3) Preference for anonymity and control over spending.

### Aim
Aim is to:
<br> (1) develop a reliable and accurate system for authenticating banknotes using ML techniques, with the objective of detecting counterfeit currency by analyzing key features such as texture, variance, skewness, and image-based properties.
<br> (2) ensure explainability of the authentication process, allowing stakeholders to understand which features contribute most to the decision-making.


|   | notebook                      | description                    |
|---|-------------------------------|--------------------------------|
|1. |[EDA](https://github.com/doscsy12/ADI_projects/blob/main/PILOT/EDA.ipynb) | EDA | 
|2. |[VGG16](https://github.com/doscsy12/ADI_projects/blob/main/PILOT/VGG16_model.ipynb) | VGG16 model |
|3. |[Resnet50](https://github.com/doscsy12/ADI_projects/blob/main/PILOT/Resnet50_model.ipynb) | Resnet50 model |
|4. |[..]()  | Xception model |
|5. |[..]()  | InceptionV3 model |

## Enhancing Voice Banking Interfaces with Pitch Recognition

### Background
As banks increasingly adopt voice-driven platforms (such as Alexa, Siri integrations), delivering a natural and reliable user experience is critical. In multilingual markets like the Philippines and Southeast Asia, tonal and intonation-rich languages present unique challenges for speech recognition. Pitch patterns play a key role in distinguishing questions, statements, and confirmations, making interactions more accurate and user-friendly.

### Aim
This pilot project aims to explore the use of pitch recognition to improve the accuracy and usability of voice banking systems, focusing on better understanding and processing of multilingual and tonal speech for enhanced customer experience. The study will experiment with different feature extraction methods to determine which best captures the relevant speech characteristics. Initial experiments will establish Logistic Regression as a baseline, followed by Support Vector Machines (SVM) as a test model.

|   | notebook                      | description                    |
|---|-------------------------------|--------------------------------|
|1. |[1st experiment](https://github.com/doscsy12/ADI_projects/blob/main/PILOT/pitch_recognition.ipynb) | Own feature extraction + LogReg and SVM models |
|2. |[2nd experiment](https://github.com/doscsy12/ADI_projects/blob/main/PILOT/pitch_recognition-pt2.ipynb)  | Feature extraction from Praat spectral analysis + LogReg and SVM models
|3. |[3rd experiment](https://github.com/doscsy12/ADI_projects/blob/main/PILOT/pitch_recognition-pt3.ipynb)  | Train CNN to extract MFCC embeddings + LogReg and SVM models  |
|4. |[4th experiment](https://github.com/doscsy12/ADI_projects/blob/main/PILOT/pitch_recognition-pt4.ipynb)  | Utilise pretrained audio embeddings from Wav2Vec2.0 (Facebook AI) + LogReg and SVM models  |
|5. |[]()  | Multi-feature / hybrid model - combine feature extraction using different techniques |
|6. |[]()  | Multi-modal model - combine features from different modalities (MFCCs, pitch, images, pretrained embeddings) | 