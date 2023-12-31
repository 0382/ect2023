import numpy as np

def matprint(mat, fmt=".4g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")      

def scale_point(x, lo, hi):
    return np.array(x) * (hi - lo) + lo
