import numpy as np
from json import JSONEncoder
import json

'''
Code from: https://pynative.com/python-serialize-numpy-ndarray-into-json/
'''
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)