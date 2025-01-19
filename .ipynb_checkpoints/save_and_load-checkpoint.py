import pickle
import numpy as np

def obj_load(file):
    print("loading",file)
    f = open(file, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def obj_save(_object, file):
    filehandler = open(file, 'wb') 
    pickle.dump(_object, filehandler)

