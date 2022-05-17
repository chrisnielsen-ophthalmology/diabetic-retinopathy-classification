# diabetic-retinopathy-classification



The purpose of this repository is to provide a system for classifying the severity of diabetic retinopathy in retinal fundus images using data from the 2015 [Kaggle EyePACS competition](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/overview). Much of this code is based on the excellent repository developed by [aladdinpersson](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Kaggles/DiabeticRetinopathy). The developed solution follows the rough outline:

1. All image data is preprocessed to remove black borders and is resized to 512 x 512 pixels
2. The mean and standard deviation of the dataset is computed to normalize the image intensity values
3. An ensemble of EfficientNet models (2xB3, 2xB4, 2xB5) are trained using MSE loss with significant data augmentation
4. Test time augmentation is used to compute the predictions for each individual model and the results are averaged to get the final ensemble predictions
5. The Quadratic Weighted Kappa metric is computed to evaluate the results



The resulting model gave a 0.814135 Kappa score on the Kaggle public test set. 





## How to use this code



### 1) Getting the data

The data can be acquired by creating a Kaggle account and downloading the train and test zip files from [here](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data). Note that the size of this dataset is quite large (88.29 GB zipped). Once the data has been downloaded, the next step is to extract the images and place them in train and test folders.



### 2) Preprocessing the data

Run the script `preprocess_images.py` to preprocess the raw images (remove border and resize). To run this script, ensure that the correct file paths to the data, and desired image resize shape are specified when calling the `fast_image_resize` function. This operation takes around an hour to run on my computer.



### 3) Computing the dataset statistics

Run the notebook `compute_dataset_statistics.ipynb` to compute the mean and standard deviation of the pixel values for all the preprocessed images. These computed statistics should be copied into the `Normalize`  function inside `utils.py` and will be used to correctly scale the image pixel intensity values to have mean 0 and standard deviation of 1 for each color channel.



### 4) Training the ensemble models

Run the code `train_ensemble.py` to train the ensemble of EfficientNet models (2xB3, 2xB4, 2xB5) using MSE loss with significant data augmentation. One important note is that for parallel processing of the DataLoader, on Windows machines it is necessary to set `persistent_workers=True`. Utility classes have been defined for each of the three architecture types (B3, B4, B5) where the number of epochs to train is uniquely specified. Roughly, training took around 10 minutes per epoch for the B3 architectures and 25 minutes per epoch for the B5 architectures on my machine. Using parallelization for the DataLoader was very important for performance. 



### 5) Compute individual model predictions using TTA

Run the code `compute_tta_all_models.ipynb` to compute the model predictions using test time augmentation (TTA). The best model epochs are chosen based off the epoch that maximizes the validation Kappa score. 



### 6) Computing final ensemble prediction

Run the code `compute_ensemble_predictions.ipynb` to combine the individual model predictions into an ensemble prediction and compute the Quadratic Weighted Kappa metric to evaluate the results. Currently this code only evaluates on the Kaggle public test set, but could easily be extended to evaluate based on the Kaggle private test set.



 

