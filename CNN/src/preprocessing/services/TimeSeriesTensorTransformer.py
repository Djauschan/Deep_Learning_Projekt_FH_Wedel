import torch
import numpy as np

'''
    PreProcessing Data Before transforming into tensor
'''
class CorrectData(object):
    def __call__(self, sample):
        return sample


'''
    Convert ndarrays in sample to Tensors.
'''
class ToTensor(object):

    def __call__(self, sample):
        arr = np.array(sample['data'])
        '''
        für 1 = channal = 1 feature
        return {
            #from 1, 10, 10 -> 1, 1, 10, 10 (in chanals added
            'x': torch.unsqueeze(torch.from_numpy(arr), 0),
            'y': torch.tensor(np.array(sample['label']))
        }        
        '''
        return {
            #from 1, 10, 10 -> 1, 1, 10, 10 (in chanals added
            'x': torch.from_numpy(arr),
            'y': torch.tensor(np.array(sample['label']))
        }
