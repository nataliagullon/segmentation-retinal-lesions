## Retinal lesions segmentation using CNNs and adversarial training

A Degree Thesis Submitted to the Faculty of Escola Tècnica d’Enginyeria de Telecomunicació de Barcelona of Universitat Politècnica de Catalunya in July 2018. In partial fulfilment of the requirements for the degree in Telecommunications Technologies and Services Engineering. 

In this project, retinal lesions (Microaneurysms, Haemorrhages, Soft Exudates and Hard Exudates) are segmented in funduscopy images using a U-Net and adversarial training.

This file contains practical information on the project implementation and how to run it.

- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Environment](#environment)
- [Running the code](#running-the-code)
  - [Preprocess data](#preprocess-data)
  - [Training](#training)
  - [Predict and Evaluate](#predict-and-evaluate)
  - [Tensorboard](#tensorboard)

## Project structure

The project has the following folder (and file) structure:

- `data/`. [After [preprocess data](#preprocess-data)]: Directory containing input data, where there is a directory for train, validation and test sets, namely `train/` , `validation/` and `test/` respectively. Inside each directory, there are all the images and ground truths corresponding to each set. Additionally, there are all the images, ground truths and masks used in this project.
- `src/`. Python files with code of the implementation.
  - `data.py`. Functions used to preprocess and prepare data (images and ground truth) to train and test the model as well as functions related to data used in other parts of the project.
  - `evaluate.py`. Functions used to evaluate the performance of the model. Results are saved inside the same directory where the predicted images are saved.
  - `model_unet.py`. Functions used to get the U-Net used in this project to segment the images.
  - `model_gan.py`. Functions used to get the networks (U-Net from `model_unet.py` and discriminator) used in the adversarial training and to combine both models.
  - `train_unet.py`. Functions used to train the U-Net model.
  - `train_gan.py`. Functions used to train the U-Net model using adversarial training. It is initialized using the weights of a pre-trained U-Net.
  - `predict.py`. Functions used to do the predictions using the U-Net.
  - `unet_generator.py`. Generator class used when training and validating the model.
  - `gan_utils.py`. Help functions that are used in the adversarial training.
  - `utils/`.
    - `losses.py`. Customized loss functions that can be used to train the model.
    - `check_data.py`. Functions used to check that the data is correctly loaded (it is run from the script `check_data.sh`).
    - `check_data.sh`. Script to structure the data as expected to run the code afterwards, as explained in [dataset](#dataset). It also checks that the data is correctly loaded.
    - `params.py`. File containing the main parameters used in the project.
- `report/`. Directory containing the report of the thesis (`report.pdf`) and the slides used in the thesis defense (`presentation.pdf`).

## Dataset

The dataset used in this project is the Indian Diabetic Retinopathy Image Dataset (IDRiD) from the [Diabetic Retinopathy: Segmentation and Grading Challenge](https://idrid.grand-challenge.org) organised at IEEE International Symposium on Biomedical Imaging (ISBI-2018).

The data can be downloaded from the [IEEEDataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid). More precisely, the dataset used is the part A named segmentation.

The default directory to locate the directories with the images and labels downloaded from the previous link is `data/` located in the main directory. In order to use the default parameters, once you donwload the `.zip` file, you can unzip it and locate it on the main directory and then, run the script located in `src/utils/check_data.sh`.

Nevertheless, if desired, you can change this location by modifying the default parameter `'data_path'` located in the file `utils/params.py` and use your own path. The preprocessed data will also be stored in this same directory.

The final structure of the directory `data/ ` should look similar to:

```
data
├── ground_truths
│   ├── EX
│   ├── HE
│   ├── MA
│   └── SE
└── images
```

## Environment

The code is implemented in Python using Keras framework running on top of TensorFlow. Therefore, the required environment for running the code and reproducing the results is a computer with a valid installation of Python 3. 

Besides that (and the built-in Python libraries), make sure to install all the requirements specified in the `requirements.txt` file. If you use the pip management system, you can just place in the root directory of this project after cloning it to your machine and type:

```
pip install -r requirements.txt
```

## Running the code

### Preprocess data

To run the main program, first of all you need to preprocess and prepare the data. To do that, the file `data.py` has to be executed from the `src` folder. It is recommended to disable the warnings due to the  low contrast of the images saved (there are a lot of them):

```
cd src/
python -W ignore data.py [options]
```

There are 2 command line arguments:

```
usage: data.py [-h] [-V]

optional arguments:
  -h, --help     show this help message and exit
  -V, --verbose  provide additional details about the program (default: False)
```

The prepared data that will be used when training and evaluating the model will be generated inside the `data` directory.

### Training

**U-Net**

To train the U-Net, you need to execute the `train_unet.py` from the `src` folder:

```
cd src/
python train_unet.py [options] -weights WEIGHTS
```

The `weights` argument is required and it indicates the name of the weights that will be saved when training: 

```
usage: train_unet.py [-h] [-V] -weights WEIGHTS -adv_training ADV_TRAINING

optional arguments:
  -h, --help            show this help message and exit
  -V, --verbose         provide additional details about the program (default:
                        False)

required arguments:
  -weights WEIGHTS      name of the weights that will be saved
```

**Adversarial training**

To train the U-Net using adversarial training, you need to execute the `train_gan.py` from the `src` folder:

```
cd src/
python train_gan.py [options] -init_weights INIT_WEIGHTS -weights WEIGHTS
```

The `init_weights` and `weights` arguments are required. The first one indicates the weights that will be loaded in the U-Net before the training and the second one indicates the name of the weights that will be saved when training using adversarial training: 

```
usage: train_gan.py [-h] [-V] -init_weights INIT_WEIGHTS -weights WEIGHTS

optional arguments:
  -h, --help            show this help message and exit
  -V, --verbose         provide additional details about the program (default:
                        False)

required arguments:
  -init_weights INIT_WEIGHTS
			initial weights to be loaded on the unet
  -weights WEIGHTS      name of the weights that will be saved
```

### Predict and Evaluate

**Prediction**

To make the predictions using the U-Net, we need to execute the `predict.py` from the `src` folder indicatins the name of the weights we want to use, using the argument `weights`. It is recommended to disable the warnings due to the  low contrast of the images saved (there are a lot of them):

```
cd src/
python -W ignore predict.py [options] -weights WEIGHTS
```

There are the following command line arguments: 

```
usage: predict.py [-h] [-V] -weights WEIGHTS

optional arguments:
  -h, --help        show this help message and exit
  -V, --verbose     provide additional details about the program (default:
                    False)

required arguments:
  -weights WEIGHTS  name of the weights to use in the prediction
```

The predicted images, as well as the computed AUC Precision-Recall measurements (in a `.txt` file), will be saved inside the directory `model/predictions_name_weights`. 

**Evaluation**

To evaluate the performance of the model, we need to execute the `evaluate.py` from the `src` folder, indicating the name of the weights we want to use as in the prediction. It is also recommended to disable the warnings due to the  low contrast of the images saved (there are a lot of them):

```
cd src/
python -W ignore evaluate.py [options] -weights WEIGHTS
```

There are the following command line arguments: 

```
usage: predict.py [-h] [-V] -weights WEIGHTS

optional arguments:
  -h, --help        show this help message and exit
  -V, --verbose     provide additional details about the program (default:
                    False)

required arguments:
  -weights WEIGHTS  name of the weights to use in the prediction
```

The results will be saved inside the directory `model/predictions_name_weights`, in a `.txt` file. Additionally, the difference maps between the predictions and ground truths will be saved in a directory called `maps`.

### Tensorboard

In order to see the evolution of the loss curves, you can run Tensorboard while your code is running to see either the live evolution or when it has already finished. In order to do so, you need to run the following command from the main directory:

```
tensorboard --logdir=model/tensorboard/
```
