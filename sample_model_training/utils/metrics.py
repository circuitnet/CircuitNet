import numpy as np

__all__ = ['corrcoef', 'mae']


def corrcoef(arr1,arr2):
    corrcoef = np.corrcoef(arr1,arr2)
    return corrcoef[0][1]

def mae(arr1,arr2):
    mae = np.sum(np.absolute(arr1-arr2)) / len(arr1)
    return mae
