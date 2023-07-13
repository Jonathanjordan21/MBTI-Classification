from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

from dataclasses import dataclass
import os,sys
import pandas as pd
import json
from utils import save_object
from src.logger import logging
from src.exception import CustomException



class ModelTrainerConfig():
    def __init__(self, train_data_path=None, test_data_path=None, data_transformation_config=None):
        self.best_trained_model_path:str=os.path.join('artifacts', 'model.pkl')
        self.best_trained_model_score:str=os.path.join('artifacts', 'model.json')
        if data_transformation_config != None:
            self.train_data_path = data_transformation_config.train_data_path
            self.test_data_path = data_transformation_config.test_data_path
        else :
            self.train_data_path = train_data_path
            self.test_data_path = test_data_path

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()
    
    def train(self):
        logging.info("Prepare for Training Model")
        models = {
            'ada_boost':AdaBoostClassifier(),
            'grad_boost':GradientBoostingClassifier(),
            'naive_bayes':MultinomialNB(),
            'SGD':SGDClassifier()
        }

        lr = [0.1,0.01,0.001, 0.0001, 0.00001]

        parameters = {
            'ada_boost':{
                'n_estimators':[50,100,150],
                'learning_rate':lr
            },
            'grad_boost':{
                'n_estimators':[50,100,150],
                'criterion':('friedman_mse', 'squared_error'),
                'max_depth':[1,3,5],
                'min_impurity_decrease':[0.,0.5,1.,1.5,2.,0.2]
            },
            'naive_bayes':{
                'alpha':[1.,0.6,0.3,0.1,1e-3,1e-6,1e-9]
            },
            'SGD' : {
                'loss':(
                    'hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 
                    'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
                ),
                'penalty':['l2', None, 'elasticnet'],
                'alpha':[1e-4, 1e-5, 1e-6, 1e-7],
                'epsilon':[0.1,0.25,0.4],
                'learning_rate':('invscaling', 'adaptive', 'optimal'),
                'average':[None, 10,30,60]
            }
        }

        for k,v in models.items():
            self.evaluate(v, parameters[k], k)
    
    def evaluate(self, model, parameters, model_name):
        try :
            train_data = pd.read_csv(self.trainer_config.train_data_path)

            logging.info(f"Start Training {model_name}...")
            clf = GridSearchCV(model, parameters)
            pipeline_clf = Pipeline([
                ('vect', CountVectorizer(stop_words='english')),
                ('tfidf', TfidfTransformer()),
                ('clf', clf)
            ])
            pipeline_clf.fit(train_data['post'], train_data['type'])

            logging.info(f"Training {model_name} Completed, Prepare for Testing")

            test_data = pd.read_csv(self.trainer_config.test_data_path)
            predicted = pipeline_clf.predict(test_data['post'], test_data['type'])

            logging.info(f"Testing {model_name} Completed. Prepare for Model Evaluation")

            f1 = f1_score(train_data['type'], predicted)
            with open(self.trainer_config.best_trained_model_score, 'r') as f:
                if os.path.exists(self.trainer_config.best_trained_model_score):
                    f1_prev = json.load(self.trainer_config.best_trained_model_score)['f1_score']
                else :
                    f1_prev = 0
            # f1_prev = 0 if f1_prev == None else f1_prev
            
            if f1 > f1_prev:
                logging.info(f"{model_name} outperforms the previous optimal model! Saving {model_name}...")
                with open(self.trainer_config.best_trained_model_score, 'w') as f:
                    json.dump({'model':clf, 'f1_score':f1}, f)
                with open(self.trainer_config.best_trained_model_path, 'wb') as f:
                    save_object(pipeline_clf, f)

            logging.info(f"Model Evaluation {model_name} Completed")
            # print("F1_Score :", f1)

            
        except Exception as err:
            CustomException(err, sys)