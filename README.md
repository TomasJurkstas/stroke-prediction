# Stroke Prediction Using Machine Learning
This project centers around the application of machine learning to train a model for categorization of individuals into two groups: those who are likely to have a stroke and those who are not. We have followed a few steps during this project:
1. Formulated the problem.
2. Performed data cleaning.
3. Conducted EDA to uncover patterns.
4. Visualized findings.
5. Applied statistical testing procedures.
6. Trained multiple ML models.
7. Analyzed model performance.
8. Selected the final model.
9. Deployed the model on google cloud.

## Data Source
We're using "Stroke Prediction Dataset" from Kaggle, which can be found [here](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

## Tools Used
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scipy
- Scikit-Learn
- Imalanced-learn
- XGBoost
- LightGBM
- Pickle
- FastAPI
- Docker
- Google Cloud Services
- Locust

## Results and Findings
- We have found that most important feature for predicting stroke is age.
- The model is very simple and has problems predicting edge/unusual cases.
- Model doesn't have good enough recall (0.77) where it could be considered deployable.
- Model was deployed on Google Cloud Services and can be interacted with [here](https://default-service-l3vhobwizq-nw.a.run.app/).

## Future Work
- Improvements on the design of the app and website could be made.
- Feature engineering could be done to allow for better performance with gradient boosted trees models.

## Usage
While not intended to, feel free to download and run the notebook on your own machine.
