'''Contains all the adjustable variables for the Neural Network'''

class Prefrences():
    
    def __init__(self):
        self.conv_stride = 2
        self.rfs = 4 #receptive field - a value of 3 represents a 3x3 window 
        self.batch_size = 100
        self.windowsize = 100
        self.data_directory = "/home/joshua/Desktop/py_projects/Project0.2/Data"
        self.proportion_training = 1
        self.n_z = 100
        self.sampling_mean = 0
        self.sampling_stddev = 1
    
        
        
