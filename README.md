# Covid19 Prediction
> This is a deep learning model using LSTM neural network to ***predict new cases of Covid19 in Malaysia*** using the past 30 days of number of cases.

# Descriptions
>The year 2020 was a catastrophic year for humanity. Pneumonia of unknown aetiology was first reported in December 2019. Since then, COVID-19 spread to 
the whole world and became a global pandemic. More than 200 countries were 
affected due to pandemic and many countries were trying to save precious lives 
of their people by imposing travel restrictions, quarantines, social distances, event 
postponements and lockdowns to prevent the spread of the virus.However, due 
to lackadaisical attitude, efforts attempted by the governments were jeopardised, 
thus, predisposing to the wide spread of virus and lost of lives. 

>The scientists believed that the absence of AI assisted automated tracking and 
predicting system is the cause of the wide spread of COVID-19 pandemic. Hence, 
the scientist proposed the usage of deep learning model to predict the daily 
COVID cases to determine if travel bans should be imposed or rescinded.

# 📙 Requirement
>> ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
>>
>>  - To run tensorboard, you need to copy your logs path and then go to cmd/terminal.
>>  - Then, type tensorboard --logdir "LOGS_PATH".(Please make sure you're in the right environment)
>>  - You may go to http://localhost:6006/ to access yout tensorboard.
![alt text](https://github.com/intan7/Covid19-Prediction/blob/main/static/run_TensorBoard.png)

# Deep Learning Model
![alt text](https://github.com/intan7/Covid19-Prediction/blob/main/static/model.png)

# TensorBoard
![alt text](https://github.com/intan7/Covid19-Prediction/blob/main/static/TensorBoard_ss.png)

# Results
The MAPE error for the model is 9.31%.

![alt text](https://github.com/intan7/Covid19-Prediction/blob/main/static/result_ss.png)

And here's the actual vs predicted graph:

![alt text](https://github.com/intan7/Covid19-Prediction/blob/main/static/actualvspredict.png)

## Powered by
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## This project is able to successfully run thanks to
 >https://github.com/MoH-Malaysia/covid19-public
