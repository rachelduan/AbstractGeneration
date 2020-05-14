from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd

def parallelize(data, function):
    '''
    processing data using multiple cores
    '''
    cores = cpu_count()
    partitions = cores

    data_split = np.array_split(data, partitions)
    pool = Pool(cores)

    data = pd.concat(pool.map(function, data_split))
    pool.close()

    pool.join()
    return data