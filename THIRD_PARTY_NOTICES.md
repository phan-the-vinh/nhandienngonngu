# THIRD-PARTY SOFTWARE AND DATA NOTICES AND INFORMATION

This project, **NhanDienCuChi: Real-Time Hand Gesture Sign Language Recognition System**, incorporates, references, or was trained on third-party software, models, and datasets under various open-source and open-data licenses. This document acknowledges the respective authors and provides the applicable license terms, copyright notices, and attribution statements.

> [!NOTE]
> This document records only third-party materials and licenses with verified provenance. It does not unilaterally assert or imply redistribution rights for datasets or models whose upstream rights are unrecorded or pending formal release.

---

## 1. Google MediaPipe Hand Landmarker

- **Project**: MediaPipe Hand Landmarker Model Bundle (`hand_landmarker.task`)
- **Copyright**: Copyright (c) Google LLC. All rights reserved.
- **License**: Apache License, Version 2.0
- **Official Source**: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- **Official Download URL**: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- **Notice**:
  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

## 2. Arda Mavi Sign Language Digits Dataset

- **Project**: Sign Language Digits Dataset
- **Authors**: Arda Mavi and Zeynep Dikle (2020)
- **Repository**: https://github.com/ardamavi/Sign-Language-Digits-Dataset
- **License**: Apache License, Version 2.0
- **Academic Citation**:
  > Mavi, A., (2020), "A New Dataset and Proposed Convolutional Neural Network Architecture for Classification of American Sign Language Digits", *arXiv:2011.08927 [cs.CV]*. Available at: https://arxiv.org/abs/2011.08927
- **Modifications**:
  The project extracted 63-dimensional MediaPipe landmark feature vectors from 170 right-hand images (10 samples each for digits 4, 6, 8 and 20 samples each for digits 0, 1, 2, 3, 5, 7, 9) to train and optimize the static digits classification model (`digits_final.joblib`).

---

## 3. ASL-HG: American Sign Language Hand Gesture Image Dataset

- **Project**: ASL-HG Dataset (Version 1)
- **Authors**: Md. Famidul Islam Pranto, Md. Rifatul Islam, Md. Ali Akbor, Nabonita Ghosh, Md. Rahatun Alam, Sudipto Chaki, Md. Masudul Islam (2023)
- **Publisher**: Mendeley Data / Data in Brief, Elsevier
- **DOI**: 10.17632/j4y5w2c8w9.1 (https://doi.org/10.17632/j4y5w2c8w9.1)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
  https://creativecommons.org/licenses/by/4.0/
- **Modifications**:
  The project extracted 63-dimensional MediaPipe landmark feature vectors from 30 image samples (10 samples each for digits 4, 6, and 8 across distinct volunteer signers) to augment the static digits classification model (`digits_final.joblib`).

---

## 4. SUBTLEX-US Word Frequency Dataset

- **Project**: SUBTLEX-US (American English Word Frequency Norms)
- **Authors**: Marc Brysbaert (Ghent University) & Boris New (Université Savoie Mont Blanc / CNRS)
- **Publisher**: Department of Experimental Psychology, Ghent University (UGent), Belgium & OpenLexicon Project
- **Source URL**: https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
  https://creativecommons.org/licenses/by-sa/4.0/
- **Academic Citation**:
  > Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English. *Behavior Research Methods*, 41(4), 977–990. DOI: 10.3758/BRM.41.4.977
- **Derivative Notice**:
  The filtered English dictionary `data/dictionary/english_words_25k.tsv` is distributed under CC BY-SA 4.0 in accordance with the ShareAlike clause. Full transformation details are provided in `data/dictionary/NOTICE.md`.

---

## 5. SigNN ASL Sign Language Alphabet Videos [J, Z]

- **Project**: ASL Sign Language Alphabet Videos [J, Z]
- **Publisher / Team**: SigNN Team
- **Source URL**: https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z
- **License as Displayed**: Creative Commons CC0 1.0 Universal (Public Domain Dedication)
  https://creativecommons.org/publicdomain/zero/1.0/
- **Notice on Personality & Privacy Rights**:
  In accordance with Section 2(d) of CC0 1.0 Universal, publicity and privacy rights of individuals appearing in video recordings are not waived. Models derived from this dataset (`model_pipeline.joblib`) are distributed strictly for academic research and experimental benchmarking purposes.

---

## 6. ASL Alphabet Dataset (Akash Nagaraj)

- **Project**: ASL Alphabet Dataset
- **Author**: Akash Nagaraj (2018)
- **Source URL**: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **License**: GNU General Public License, Version 2.0 (GPL-2.0)
  https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
- **Notice**:
  A subset of 45,236 static letter images from this dataset was utilized for feature extraction and research benchmarking in the static letters branch (`uvw_medium_model.joblib`).

