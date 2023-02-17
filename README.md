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
* [AC-GAN Training and Generation](#ac-gan-training-and-generation)
* [AC-GAN Evaluation: FID, Intra FID, Inception Score (IS), t-SNE](#ac-gan-evaluation-fid-intra-fid-inception-score-is-t-SNE)
* [Chest X-Ray Classification: Pneumonia and COVID-19 detection with GAN augmentation](#chest-x-ray-classification-pneumonia-and-covid-19-detection-with-gan-augmentation)
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

## AC-GAN Training and Generation

1. Run the code and follow the detailed instructions in `AC-cGAN-training.ipynb` to perform AC-GAN training.
2. Generates images in large quantities using `AC-cGAN-generator.ipynb`.

## AC-GAN Evaluation: FID, Intra FID, Inception Score (IS), t-SNE

Run the code and follow the detailed instructions in `AC-cGAN-evaluate.ipynb` to:
- Displaying training metrics (**Losses and Accuracies**)
- Calculate **Fréchet inception distance (FID)**
- Calculate the **Intra Fréchet inception distance (Intra FID)**
- Calculate the **Inception Score (IS)**
- View **t-SNE two-dimensional embeddings**

## Chest X-Ray Classification: Pneumonia and COVID-19 detection with GAN Augmentation

Use the notebooks in `CXR Classification` folder to train and evaluate classification models for the detection of COVID-19, Pneumonia and Absence of symptoms in Chest X-ray. Some forms of data augmentation are tested, including generation by trained AC-GAN. 

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
