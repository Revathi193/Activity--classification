from surround import Estimator, SurroundData, Validator
import pandas as pd
import numpy as np
from numpy import mean
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')


class ActivitypredictionData(SurroundData):
    input_data = None
    output_data = None


class ValidateData(Validator):
    def validate(self, surround_data, config):
        pass

class Main(Estimator):
    def estimate(self, surround_data, config):
        print("ACTIVITY CLASSIFICATION/PREDICTION")
        predict_data = surround_data.input_data
        df_pred=pd.DataFrame(predict_data)
        df_pred=df_pred.T
        df_pred.columns=['acc_x','acc_y','acc_z']
        df_pred['acc_x'] = df_pred['acc_x'].astype(int)
        df_pred['acc_y'] = df_pred['acc_y'].astype(int)
        df_pred['acc_z'] = df_pred['acc_z'].astype(int)
        ## bucketing x_acceleration
        accx_bins=[0,300,700,1000,1500,2000,2400,2800,3400,3600,4000]
        accx_labels=['0 - 300','300-700', '700-1000', '1000-1500','1500-2000','2000-2400','2400-2800','2800-3400','3400-3600','3600+']
        df_pred['accx_bucket'] = pd.cut(df_pred.acc_x, accx_bins, labels = accx_labels,include_lowest = True)
        ## bucketing y_acceleration
        accy_bins=[0,20,100,200,250,300,700,1000,1500,2000,2400,2800,3400,3600,4000,4500]
        accy_labels=['0 - 20','20-100','100-200', '200-250', '250-300',
                                '300-700', '700-1000', '1000-1500','1500-2000','2000-2400','2400-2800','2800-3400','3400-3600','3600-4000','4000+']
        df_pred['accy_bucket'] = pd.cut(df_pred.acc_y, accy_bins, labels = accy_labels,include_lowest = True)
        ## bucketing z_acceleration
        accz_bins=[0,20,200,300,700,1000,1500,2000,2400,2800,3400,3600,4000,4500]
        accz_labels=['0 - 20','20-200','200-300',
                                '300-700', '700-1000', '1000-1500','1500-2000','2000-2400','2400-2800','2800-3400','3400-3600','3600-4000','4000+']
        df_pred['accz_bucket'] = pd.cut(df_pred.acc_z, accz_bins, labels = accz_labels,include_lowest = True)
        print(df_pred)
        df_cat_pred=df_pred[['accx_bucket','accy_bucket','accz_bucket']]
        print(df_cat_pred)
        for col in df_cat_pred.columns:
            df_cat_pred[col] = df_cat_pred[col].astype('category')
            df_cat_pred[col] = df_cat_pred[col].cat.add_categories('Unknown')
            df_cat_pred[col].fillna('Unknown', inplace =True)
        dict_all_loaded = pickle.load(open("data/dict_all.obj", 'rb'))
        print(dict_all_loaded)
        df_cat_pred['accx_bucket']=df_cat_pred['accx_bucket'].map(dict_all_loaded['accx_bucket'])
        df_cat_pred['accy_bucket']=df_cat_pred['accy_bucket'].map(dict_all_loaded['accy_bucket'])
        df_cat_pred['accz_bucket']=df_cat_pred['accz_bucket'].map(dict_all_loaded['accz_bucket'])
        print(df_cat_pred)
        for col in df_cat_pred.columns:
            df_cat_pred[col] = pd.to_numeric(df_cat_pred[col])
        print(df_cat_pred)
        # loading pickle files
        model_xgb_1 = pickle.load(open("models/working_comp.pkl", 'rb'))
        model_xgb_2 = pickle.load(open("models/stand_walk_stair.pkl", 'rb'))
        model_xgb_3 = pickle.load(open("models/standing.pkl", 'rb'))
        model_xgb_4 = pickle.load(open("models/walking.pkl", 'rb'))
        model_xgb_5 = pickle.load(open("models/going_stairs.pkl", 'rb'))
        model_xgb_6 = pickle.load(open("models/walking_talking.pkl", 'rb'))
        model_xgb_7 = pickle.load(open("models/talk_stand.pkl", 'rb'))

        #probabilities
        activity_1 = pd.DataFrame(model_xgb_1.predict_proba(df_cat_pred))
        activity_2 = pd.DataFrame(model_xgb_2.predict_proba(df_cat_pred))
        activity_3 = pd.DataFrame(model_xgb_3.predict_proba(df_cat_pred))
        activity_4 = pd.DataFrame(model_xgb_4.predict_proba(df_cat_pred))
        activity_5 = pd.DataFrame(model_xgb_5.predict_proba(df_cat_pred))
        activity_6 = pd.DataFrame(model_xgb_6.predict_proba(df_cat_pred))
        activity_7 = pd.DataFrame(model_xgb_7.predict_proba(df_cat_pred))

        df_pred['Working at Computer'] = activity_1[1].values
        df_pred['Standing Up, Walking and Going up\down stairs'] = activity_2[1].values
        df_pred['Standing'] = activity_3[1].values
        df_pred['Walking'] = activity_4[1].values
        df_pred['Going Up\Down Stairs'] = activity_5[1].values
        df_pred['Walking and Talking with Someone'] = activity_6[1].values
        df_pred['Talking while Standing'] = activity_7[1].values

        output=df_pred[['Working at Computer','Standing Up, Walking and Going up\down stairs','Standing',
                        'Walking','Going Up\Down Stairs','Walking and Talking with Someone','Talking while Standing']]
        output=output.sort_values(by=0, ascending=False, axis=1)
        output=output.iloc[:, : 5]
        print(output)
        out=output.to_dict('records')
        surround_data.output_data = out


    def fit(self, surround_data, config):
        print("TODO: MODEL TRAINING & BUILDING")
        df=surround_data.input_data
        #Check for missing values
        def missing_data(df):

                    rec = []
                    for column_name in df.columns:
                        miss_count = df[column_name].isnull().sum(axis=0)
                        miss_percent=miss_count/df.shape[0]
                        rec.append([column_name,miss_count,miss_percent*100])
                        df_stats = pd.DataFrame.from_records(rec, columns = ['column name',
                                                                             'missing_count','missing_percentage'])
                        df_stats = df_stats.sort_values('missing_percentage',axis=0, ascending = False )
                    return df_stats;
        df_missing = missing_data(df)
        print("Missing Values \n",df_missing)
        ## bucketing x_acceleration
        accx_bins=[0,300,700,1000,1500,2000,2400,2800,3400,3600,4000]
        accx_labels=['0 - 300','300-700', '700-1000', '1000-1500','1500-2000','2000-2400','2400-2800','2800-3400','3400-3600','3600+']
        df['accx_bucket'] = pd.cut(df.acc_x, accx_bins, labels = accx_labels,include_lowest = True)
        ## bucketing y_acceleration
        accy_bins=[0,20,100,200,250,300,700,1000,1500,2000,2400,2800,3400,3600,4000,4500]
        accy_labels=['0 - 20','20-100','100-200', '200-250', '250-300',
                                '300-700', '700-1000', '1000-1500','1500-2000','2000-2400','2400-2800','2800-3400','3400-3600','3600-4000','4000+']
        df['accy_bucket'] = pd.cut(df.acc_y, accy_bins, labels = accy_labels,include_lowest = True)
        ## bucketing z_acceleration
        accz_bins=[0,20,200,300,700,1000,1500,2000,2400,2800,3400,3600,4000,4500]
        accz_labels=['0 - 20','20-200', '200-300',
                                '300-700', '700-1000', '1000-1500','1500-2000','2000-2400','2400-2800','2800-3400','3400-3600','3600-4000','4000+']
        df['accz_bucket'] = pd.cut(df.acc_z, accz_bins, labels = accz_labels,include_lowest = True)
        # df.iloc[-1] = np.nan
        ## categorical encoding
        df_cat=df[['accx_bucket','accy_bucket','accz_bucket','activity']]
        for col in df_cat.columns:
            df_cat[col] = df_cat[col].astype('category')
            df_cat[col] = df_cat[col].cat.add_categories('Unknown')
            df_cat[col].fillna('Unknown', inplace =True)
        df_final=df[['accx_bucket','accy_bucket','accz_bucket','activity']]
        le = LabelEncoder()
        print('df_cat.columns',df_cat.columns)
        dict_all = dict(zip([], []))
        for col in df_cat.columns:
            temp_keys = df_cat[col].values
            print('df_cat[col]',df_cat[col])
            temp_values = le.fit_transform(df_cat[col])
            print("done",temp_values)
            dict_temp = dict(zip(temp_keys, temp_values))
            dict_all[col] = dict_temp
            print("Encoder---->",dict_all)

        ##Label mapping for final table##################
        for col in df_cat.columns:
            df_final.replace(dict_all[col], inplace=True)

        pickle.dump(dict_all, open("data/dict_all.obj", 'wb'))
        for col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col])
        df_final['seq_no']=df['seq_no']
        df_final = df_final[df_final.activity != 0]
        df_model=df_final.drop(['seq_no'],axis=1)
        ## feature matrix and target variable
        X=df_model[['accx_bucket','accy_bucket','accz_bucket']]
        X_var=X
        y=df_model['activity']
        df_processed = pd.get_dummies(df_model['activity'])
        df_processed.head()
        ## each activity as class
        df_final['Working at Computer'] = df_processed[1]
        df_final['Standing Up, Walking and Going up\down stairs'] = df_processed[2]
        df_final['Standing'] = df_processed[3]
        df_final['Walking'] = df_processed[4]
        df_final['Going Up\Down Stairs'] = df_processed[5]
        df_final['Walking and Talking with Someone'] = df_processed[6]
        df_final['Talking while Standing'] = df_processed[7]
        def activity_one(df):
            y_prod1 =df_final['Working at Computer']
            print("Value Counts",y_prod1.value_counts())
            X_train_1,X_test_1,y_train_1,y_test_1=train_test_split(X_var,y_prod1,test_size=0.10,random_state=0)
            xgb = XGBClassifier(objective ='multi:softprob', learning_rate = 0.2,num_class=2)
            model_xgb_1=xgb.fit(X_train_1, y_train_1)
            prediction_xgb= model_xgb_1.predict(X_test_1)
            print('\n Accuracy: ',accuracy_score(y_test_1, prediction_xgb))
            return model_xgb_1;
        def activity_two(df):
            y_prod2 =df_final['Standing Up, Walking and Going up\down stairs']
            print("Value Counts",y_prod2.value_counts())
            X_train_2,X_test_2,y_train_2,y_test_2=train_test_split(X_var,y_prod2,test_size=0.20,random_state=0)
            xgb_2 = XGBClassifier(objective ='multi:softprob', learning_rate = 0.2,
                                        max_depth = 20, alpha = 20, n_estimators = 20,num_class=2)
            model_xgb_2=xgb_2.fit(X_train_2, y_train_2)
            prediction_xgb= model_xgb_2.predict(X_test_2)
            print('\n Accuracy: ',accuracy_score(y_test_2, prediction_xgb))
            return model_xgb_2;
        def activity_three(df):
            y_prod3 =df_final['Standing']
            X_train_3,X_test_3,y_train_3,y_test_3=train_test_split(X_var,y_prod3,test_size=0.1,random_state=0)
            xgb_3 = XGBClassifier(objective ='multi:softprob', learning_rate = 0.3,
                max_depth = 30, alpha = 30, n_estimators = 30,num_class=3)
            model_xgb_3=xgb_3.fit(X_train_3, y_train_3)
            prediction_xgb= model_xgb_3.predict(X_test_3)
            print('\n Accuracy: ',accuracy_score(y_test_3, prediction_xgb))
            return model_xgb_3;
        def activity_four(df):
            y_prod4 =df_final['Walking']
            X_train_4,X_test_4,y_train_4,y_test_4=train_test_split(X_var,y_prod4,test_size=0.10,random_state=0)
            xgb_4 = XGBClassifier(objective ='multi:softprob', learning_rate = 0.4,
                max_depth = 40, alpha = 40, n_estimators = 40,num_class=4)
            model_xgb_4=xgb_4.fit(X_train_4, y_train_4)
            prediction_xgb= model_xgb_4.predict(X_test_4)
            print('\n Accuracy: ',accuracy_score(y_test_4, prediction_xgb))
            return model_xgb_4;
        def activity_five(df):
            y_prod5 =df_final['Going Up\Down Stairs']
            X_train_5,X_test_5,y_train_5,y_test_5=train_test_split(X_var,y_prod5,test_size=0.10,random_state=0)
            xgb_5 = XGBClassifier(objective ='multi:softprob', learning_rate = 0.5,
                            max_depth = 50, alpha = 50, n_estimators = 50,num_class=5)
            model_xgb_5=xgb_5.fit(X_train_5, y_train_5)
            prediction_xgb= model_xgb_5.predict(X_test_5)
            print('\n Accuracy: ',accuracy_score(y_test_5, prediction_xgb))
            return model_xgb_5;
        def activity_six(df):
            y_prod6 =df_final['Walking and Talking with Someone']
            X_train_6,X_test_6,y_train_6,y_test_6=train_test_split(X_var,y_prod6,test_size=0.10,random_state=0)
            xgb_6 = XGBClassifier(objective ='multi:softprob', learning_rate = 0.6,
                max_depth = 60, alpha = 60, n_estimators = 60,num_class=6)
            model_xgb_6=xgb_6.fit(X_train_6, y_train_6)
            prediction_xgb= model_xgb_6.predict(X_test_6)
            print('\n Accuracy: ',accuracy_score(y_test_6, prediction_xgb))
            return model_xgb_6;
        def activity_seven(df):
            y_prod7 =df_final['Talking while Standing']
            X_train_7,X_test_7,y_train_7,y_test_7=train_test_split(X_var,y_prod7,test_size=0.10,random_state=0)
            xgb_7 = XGBClassifier(objective ='multi:softprob', learning_rate = 0.7,
                max_depth = 70, alpha = 70, n_estimators = 70,num_class=7)
            model_xgb_7=xgb_7.fit(X_train_7, y_train_7)
            prediction_xgb= model_xgb_7.predict(X_test_7)
            print('\n Accuracy: ',accuracy_score(y_test_7, prediction_xgb))
            return model_xgb_7;
        #creating pickle files for new contact
        filename_1 = 'models/working_comp.pkl'
        filename_2 = 'models/stand_walk_stair.pkl'
        filename_3 = 'models/standing.pkl'
        filename_4 = 'models/walking.pkl'
        filename_5 = 'models/going_stairs.pkl'
        filename_6 = 'models/walking_talking.pkl'
        filename_7 = 'models/talk_stand.pkl'
        #saving pickle files for existing contact
        pickle.dump(activity_one(df), open(filename_1, 'wb'))
        pickle.dump(activity_two(df), open(filename_2, 'wb'))
        pickle.dump(activity_three(df), open(filename_3, 'wb'))
        pickle.dump(activity_four(df), open(filename_4, 'wb'))
        pickle.dump(activity_five(df), open(filename_5, 'wb'))
        pickle.dump(activity_six(df), open(filename_6, 'wb'))
        pickle.dump(activity_seven(df), open(filename_7, 'wb'))
