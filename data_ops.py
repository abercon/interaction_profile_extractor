'''This module contains the functions required for data import and manipulation'''

from pandas import read_csv
import pathlib
from os import fspath
import random
from prefrences import Prefrences


class DataStore():
    '''Creates a DataStore object which has 2 public objects: train and test
    and methods for feeding batches to the Neural Network(NN).
    These contain the data used by the NN to learn and to 
    test how well the NN has learned. Prefrences regarding how the data should 
    be handled are contained within the prefrences object which can be altered
    by manipulating the variables under the __init__ function of the prefrences
    module.'''
    
    class _data_repo():
        '''A subclass that contains the methods required for feeding batches to 
        the NN and tracking the batch number'''
        
        def __init__(self, data):
            self.batch_number = 0
            self.data = data
            
        def get_next_batch(self):
            return self.data[self.batch_number]
            self.batch_number += 1

    def __init__(self):
        self.pref = Prefrences()
        data_root = pathlib.Path(self.pref.data_directory)
        data_dirs = [fspath(x) for x in data_root.iterdir() if x.is_dir()]

        data = self._get_data(data_dirs, self.pref.windowsize)
        training_data, test_data = self._assign_data(data, self.pref.proportion_training, self.pref.batch_size)
        self.test = self._data_repo(test_data)
        self.train = self._data_repo(training_data)        
    
    def _get_window(self, data, size, position):
        '''Extracts a square window of a given size at a given position and 
        returns it.'''
        window = (data[int(position[0]):int(position[0]+size),
                       int(position[1]):int(position[1]+size)])
        return window

    def _get_windows(self, data, size):
        '''Extracts consecutive overlapping windows of a given size along the 
        diagonal of a heatmap matrix'''
        return [self._get_window(data, size, [i,i]) for i in range(0, len(data))]
    
    def _get_data(self, data_dirs, window_size):
        '''Gets all the windows from all heatmaps and flattens them into a
        single list'''
        raw_data = [read_csv(data_dirs[i]) for i in data_dirs]
        processed_data = [self._get_windows(raw_data[i], window_size) for i in 
                          raw_data] #removed self.data
        return [item for sublist in processed_data for item in sublist]
    
    def _batch_data(self, data, batch_size):
        '''splits a dataset data into smaller batches of size batch_size'''
        batches = [data[x:x+batch_size] for x in range(0, len(data), batch_size)]
        return batches
        
    
    def _assign_data(self, data, proportion, batch_size):
        '''psuedo-randomly assigns the data to two groups by percentage, the
        proportion variable directly specifies the amount of data in set 2. 
        Returns the results as a batched tuple.'''
        set1 = data
        set2 = []
        n = len(data)
        for i in range(0, int(len(data)*proportion)):
            print (n)
            random_value = random.randrange(0, n)
            print (random_value)
            set2.append(set1.pop(random_value))
            n-=1
        return self._batch_data(set1, self.pref.batch_size), self._batch_data(set2, self.pref.batch_size)