Activity Classification

The aim is to create an ML model that can accept raw values from a 3 DoF
sensor and classify up to 5 different activities based on the raw data.

Relevant Information:

 --- The dataset collects data from a wearable accelerometer mounted on
the chest

 --- Sampling frequency of the accelerometer: 52 Hz

 --- Accelerometer Data are Uncalibrated

 --- Number of Participants: 15

 --- Number of Activities: 7

 --- Data Format: CSV

Dataset Information

 --- Data are separated by participant

 --- Each file contains the following information

 ---- sequential number, x acceleration, y acceleration, z acceleration,
label

 --- Labels are codified by numbers

 --- 1: Working at Computer

 --- 2: Standing Up, Walking and Going up\\down stairs

 --- 3: Standing

 --- 4: Walking

 --- 5: Going Up\\Down Stairs

 --- 6: Walking and Talking with Someone

 --- 7: Talking while Standing

 Approach Taken:

 • The data from 15 participants were taken into 15 different dataframes
and concatenated to single dataframe.

 • The sequentail number is replaced by corresponding participant no.

 • An initial analyis were done in Jupyter Notebook regarding the counts
of each column, checking for missing values and the analyis were plotted
as Histogram and Scatter plot.

 • The data preprocessing includes, since the x acceleration, y
acceleration and z acceleration have number of values and building model
with such values cannot yield good results, their values are needed to
be converted to bins. These needs to be further label encoded.

 • ML model can build by using x,y and z acceleration bins as feature
matrix and activity as the target variable.

 • Since there are 7 different activities, 7 ML models are built using
XGBoost as the Classifier. XGBoost is used since this it has ability to
give probabilistic predictions.

 • The project is done in Surround framework.

 • Prediction can be done using APIs in Postman.

 • For that Sample Body=\["2500","2500","1000"\] which are x
acceleration, y acceleration and z acceleration values respectively
coming from 3 DoF Sensor.

 • These values where taken and converted to a dataframe and then
preprocessing is done as similar to training part.

 • Using the 7 models, probability of predicting each activity can be
find for this set of sensor values.

 • Based on these values, top 5 activities that have maximum values will
be taken.

 • Sample response:

 {

"Activity Classification": \[

{

"Working at Computer": 0.9997205138206482,

"Standing Up, Walking and Going up\\\\down stairs":
0.018132194876670837,

"Standing": 0.0016368260839954019,

"Walking and Talking with Someone": 0.0008592443773522973,

"Walking": 0.000809031946118921

}

\]

}

 • Dockerfile, docker-compose.yml and .gitlab-ci.yml are also included
which are a part of CI/CD deployment.

 • Image name given: activityprediction:latest

 • Port taken: 8080

Run Commands:

 • Training: python3 -m activityprediction --mode train

 • Prediction: python3 -m activityprediction --mode web
