# CXR-ACGAN: Auxiliary Classifier GAN (AC-GAN) for Conditional Generation of Chest X-Ray Images (Pneumonia, COVID-19 and normal patients).

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

## Table of contents
* [Abstract](#abstract)
* [Report](https://www.slideshare.net/Giorgio469575/cxracgan-auxiliary-classifier-gan-for-conditional-generation-of-chest-xray-images-pneumonia-covid19-and-healthy-patients-255904534/Giorgio469575/cxracgan-auxiliary-classifier-gan-for-conditional-generation-of-chest-xray-images-pneumonia-covid19-and-healthy-patients-255904534)
* [Requirements](#requirements)
* [COVIDx CXR-3: Dataset and Image Pre-processing](#covidx-cxr-3-dataset-and-image-pre-processing)
* [Extreme Extractive Text Summarization](#extreme-extractive-summarization-task)
* [Topic Modeling](#topic-modeling-task)
* [Status](#status)
* [Contact](#contact)
* [License](#license)
* [Contributing](#contributing)

## Abstract

The objective of this project is to train an auxiliary classifier GAN (AC-GAN) to generate chest X-rays of healthy patients, COVID-19 patients, and non-COVID-19 pneumonia patients. Additionally, we use the GAN for data augmentation on the unbalanced COVIDx dataset to balance the minority classes and improve the performance of classifiers. The resulting generative model will enable the synthesis of conditionally generated chest X-rays, with potential applications in medical research and diagnostics. The GAN is trained directly on COVIDx CXR-3 dataset with simple preprocessing and positional data augmentation.

## Requirements

- python 3.8
- ipython
- ipykernel
- matplotlib
- pandas
- seaborn
- tqdm==4.64.1
- scikit-learn==1.2.0
- glob2==0.7
- keras==2.10.0
- Keras-Preprocessing==1.1.2
- numpy==1.24.2
- opencv_python==4.7.0.68
- pandas==1.5.3
- Pillow==9.4.0
- imageio==2.25.0

## COVIDx CXR-3: Dataset and Image Pre-processing

### Step 1. Download and extract the dataset

Download COVID-x CXR-3 from the [official Kaggle Folder](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2?select=competition_test) and extract `train` and `test` folders in `./Data/COVIDx/`, resulting in a folder tree like this:

```
project_folder
└───Data
    ├───COVIDx
    |    ├───test_COVIDx9A.txt
    |    ├───train_COVIDx9A.txt
    |    ├───train
    |    └───test
    ├───COVIDx-splitted-resized-112
    ├─── ...
```

### Step 2. Perform class splitting of the dataset
Run the `./Data/train_test_class_split.sh` bash script which will subdivide the training and testing images within the COVIDx-splitted-resized-112 folder according to their class. So that we can then use `flow_from_directory`. You get a directory tree like this:
```
project_folder
└───Data
    ├───COVIDx
    ├───COVIDx-splitted-resized-112
    |    ├───test_COVIDx9A.txt
    |    ├───train_COVIDx9A.txt
    |    ├───train
    |    |   ├───COVID-19
    |    |   ├───normal
    |    |   └───pneumonia
    |    └───test
    ├─── ...
```


### Step 3. (Opt) Perform images resizing to 112 x 122
This step is intended to reduce the size of the dataset. It is not essential because resizing is also performed in the training phase. Run the `./Data/resize_all.py` script.

## AC-CGAN Training

### Step 0. Split and clean 'ProcessedData' for easy management
Run notebook `1_preprocessing4summarization.ipynb` in order to:
- remove document without summary
- remove document with a single sentence
- split train dataset 

```
project_folder
└───Processed Data For Summarization
    ├───test_0.json
    ├───test_1.json
    ├───test_2.json
    ├───train_1_0.json
    ├───train_1_1.json
    ├───train_1_2.json
    ├───train_2_0.json
    ├───train_2_1.json
    ├───train_2_2.json
    ├───  ...
    ├───train_8_0.json
    ├───train_8_1.json
    ├───train_8_2.json
    ├───train_9_0.json
    ├───val_0.json
    ├───val_1.json
    └───val_2.json
```

### Step 1. Create a feature matrix for each of the JSON in 'Processed Data For Summarization'
Run `1_featureMatrixGeneration.py` obtaining feature matrices (sentences x features). You get a directory tree like this:

```
project_folder
└───Feature Matrices
    ├───test_0.csv
    ├───test_1.csv
    ├───test_2.csv
    ├───train_1_0.csv
    ├───train_1_1.csv
    ├───train_1_2.csv
    ├───train_2_0.csv
    ├───train_2_1.csv
    ├───train_2_2.csv
    ├───  ...
    ├───train_8_0.csv
    ├───train_8_1.csv
    ├───train_8_2.csv
    ├───train_9_0.csv
    ├───val_0.csv
    ├───val_1.csv
    └───val_2.csv
```
    
 Run the notebook `1_featureMatrixGeneration2.ipynb` to join train, val and test datasets. You get a directory tree like this:
 
 ```
 project_folder
 └───Feature Matrices
    ├───test_0.csv
    ├───test_1.csv
    ├───test_2.csv
    ├───train_1_0.csv
    ├───train_1_1.csv
    ├───train_1_2.csv
    ├───train_2_0.csv
    ├───train_2_1.csv
    ├───train_2_2.csv
    ├───  ...
    ├───train_8_0.csv
    ├───train_8_1.csv
    ├───train_8_2.csv
    ├───train_9_0.csv
    ├───val_0.csv
    ├───val_1.csv
    ├───val_2.csv
    ├───test.csv
    ├───train.csv
    └───val.csv
```
    
Features generated at this step are the following:
- sentence_relative_positions
- sentence_similarity_score_1_gram
- word_in_sentence_relative
- NOUN_tag_ratio
- VERB_tag_ratio
- ADJ_tag_ratio
- ADV_tag_ratio
- TF_ISF
 
    
### Step 2. Perform CUR undersampling
Run notebook `1_featureMatrixUndersampling.ipynb` in order to perform CUR undersampling on both train and validation data sets. You get a directory tree like this:

```
project_folder
└───Undersampled Data
    ├───trainAndValMinorityClass.csv
    └───trainAndValMajorityClassUndersampled.csv
```

Majority and minority class are splitted because CUR undersampling works only on the majority class

### Step 3. Perform EditedNearestNeighbours(ENN) undersamplig
Run notebook `1_featureMatrixAnalysis.ipynb` to perform EEN undersampling. You get a directory tree like this:

```
project_folder
└───Undersampled Data
    ├───trainAndValUndersampledENN3.csv
    ├───trainAndValMinorityClass.csv
    └───trainAndValMajorityClassUndersampled.csv
```

### Step 4. Machine Learning model selection and evaluation
Run notebook `1_featureMatrixAnalysis.ipynb` to perform a RandomizedSearcCV over the following models
- RandomForestClassifier
- LogisticRegression
- HistGradientBoostingClassifier

with a few possible parameters configuration. 

Then, evaluate the resulting best model on the test set with respect to:
- ROC curve
- Recall
- Precision
- Accuracy

### Step 5. Perform Maximal Marginal Relevance(MMR) selection
Run notebook `1_featureMatrixAnalysis.ipynb` to perform MMR and obtain an extractive summary for each document in the test set.

### Step 6. Summary Evaluation
Run notebook `1_featureMatrixAnalysis.ipynb` to measure summaries quality by means of 
- Rouge1
- Rouge2 
- RougeL 

## Topic Modeling task
### Step 0. Perform preprocessing
Run the `2_preprocessing4topic_modeling.ipynb` script to process and extract only the useful data. The output is saved here:

```
project_folder
└───processed_dataset
    ├───test.json
```

### Step 1. Perform topic modeling on the test set
Run the `2_topic_modeling.ipynb` script which will perform LDA (with grid search of the best hyper-parameters) and LSA. The script saves 9 CSV files, 3 for LSA and 6 for LDA (UMass and CV coherence measures), containing: document-topic matrix, topic-term matrix and a table with topic insights.

```
project_folder
└───Results_topic_modeling
    ├───lda_doc_topic.csv
    ├───lda_doc_topic_CV.csv
    ├───lda_top_terms.csv
    ├───lda_top_terms_CV.csv
    ├───lda_topic_term.csv
    ├───lda_topic_term_cv.csv
    ├───lsa_doc_topic.csv
    ├───lsa_top_terms.csv
    ├───lsa_topic_term.csv
```
Saves images regarding the number of words per document and wordcloud in

```
project_folder
└───Images
```

Saves hyperparameters grid search results for UMass and CV coherence in
```
project_folder
└───Hyperparameters
    ├───tuning.csv
    ├───tuning_CV.csv
```

## Status

 Project is: ![##c5f015](https://via.placeholder.com/15/c5f015/000000?text=+)  _Done_


## Contact

[Giorgio Carbone](https://github.com/giocoal) - feel free to contact me!


## License
* >You can check out the full license [here](https://github.com/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator/blob/main/README.md)

This project is licensed under the terms of the **MIT** license.

## Contributing

1. Fork it (<https://github.com/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator.git>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request


# Contributors

* [Giorgio Carbone](https://github.com/giocoal)
* [Marco Scatassi](https://github.com/marco-scatassi)
* [Gianluca Scuri](https://github.com/gianscuri)    

<!-- Project is: ![##c5f015](https://via.placeholder.com/15/c5f015/000000?text=+)  _Done_
 Project is: ![##ff0000](https://via.placeholder.com/15/ff0000/000000?text=+)  _Under-Proccess_

[![Build](https://github.com/SimonIT/spotifylyrics/workflows/Build/badge.svg)](https://github.com/SimonIT/spotifylyrics/actions?query=workflow%3ABuild)
[![Current Release](https://img.shields.io/github/release/SimonIT/spotifylyrics.svg)](https://github.com/SimonIT/spotifylyrics/releases)
[![License](https://img.shields.io/github/license/SimonIT/spotifylyrics.svg)](https://github.com/SimonIT/spotifylyrics/blob/master/LICENSE)
[![GitHub All Releases](https://img.shields.io/github/downloads/SimonIT/spotifylyrics/total)](https://github.com/SimonIT/spotifylyrics/releases)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator.svg?style=for-the-badge
[contributors-url]: https://github.com/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator.svg?style=for-the-badge
[forks-url]: https://github.com/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator/network/members
[stars-shield]: https://img.shields.io/github/stars/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator.svg?style=for-the-badge
[stars-url]: https://github.com/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator/stargazers
[issues-shield]: https://img.shields.io/github/issues/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator.svg?style=for-the-badge
[issues-url]: https://github.com/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator/issues
[license-shield]: https://img.shields.io/github/license/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator.svg?style=for-the-badge
[license-url]: https://github.com/giocoal/CXR-ACGAN-chest-Xray-COVID-19-pneumonia-generator/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/giorgio-carbone-63154219b/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
