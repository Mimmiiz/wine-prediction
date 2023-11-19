# Wine Quality Prediciton
This lab included predicting the wine quality from the [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality). The goal is to predict the quality of the wine. The dataset includes wine qualities ranging from 3 to 9.

The goals of the lab includes:
1. Clean up the data, feature engineering.
2. Write a feature pipeline notebook that registers the wine quality dataset as a Feature
Group with Hopsworks.
3. Write a training pipeline that reads training data with a Feature View from Hopsworks,
trains a regression or classifier model to predict if a wine’s quality. Register the model
with Hopsworks.
4. Write a Gradio application that downloads your model from Hopsworks and provides a User Interface to allow users to enter or select feature values to predict the quality of a wine for the features you entered.
5. Write a synthetic wine generator function and write a new “daily” feature pipeline that runs once per day to add a new synthetic wine.
6. Write a batch inference pipeline to predict the quality of the new wine(s) added, and build a Gradio application to show the most recent wine quality prediction and outcome, and a confusion matrix with historical prediction performance. 

## Feature engineering and modeling
To predict the quality of wines, some features were removed since they seemed to not contribute much to the quality of wine. The features that were kept include: type, volatile_acidity, chlorides, density, sulphates, and alcohol. 

The model that was used was the k-nearest neighbors. The model did not achieve high accuracy, especially for high and low quality wines. There was not much training data, especially for higher quality wines, additionally there were no red wines with a quality of 9 or higher. 

## Links to the UIs
Link to Huggingface Space where you can enter feature values to predict wine quality:
[https://huggingface.co/spaces/Mimmiiz/wine](https://huggingface.co/spaces/Mimmiiz/wine)

Link to Huggingface Space where you can view the most recent wine added to the Feature Store and the predicted quality of the wine:
[https://huggingface.co/spaces/Mimmiiz/wine-monitor](https://huggingface.co/spaces/Mimmiiz/wine-monitor). Currently the confusion matrix is missing since the model has a hard time predicting low and high quality wines. 