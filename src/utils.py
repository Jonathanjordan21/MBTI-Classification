import pickle,sys
from src.exception import CustomException
def save_object(obj, path):
    try :
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as err:
        CustomException(err, sys)

def load_object(path):
    try:
        obj = None
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except Exception as err:
        CustomException(err, sys)