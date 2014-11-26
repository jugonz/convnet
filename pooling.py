from layer import Layer
import numpy as np

class PoolingLayer(Layer):
    def __init__(self, numMaps, winSize, stride, type='max'):
        self.numMaps = numMaps
        self.winSize = winSize
        self.stride = stride
        self.type = type
        self.poolFunc = {'max': self._max_pool, 'mean': self._mean_pool}
        assert(self.stride <= self.winSize)
        
    def forward_prop(self, maps):
        assert(self.numMaps == maps.shape[0])
        assert(maps.shape[1] == maps.shape[2])
        
        mapW = maps.shape[1]
        poolW = (mapW/self.winSize) + ((mapW/self.winSize) - 1)*((self.winSize/self.stride) - 1)
        pooled = np.zeros((self.numMaps, poolW, poolW))
        
        for m in xrange(self.numMaps):
            for i in xrange(poolW):
                for j in xrange(poolW):
                    pooled[m,i,j] = self.poolFunc[self.type](maps(m, i*self.stride:i*self.stride+self.winSize, j*self.stride:j*self.stride+self.winSize))
                    
        return pooled
        
    def backward_prop(self, inp, error):
        pass
        
    def _mean_pool(self, window)
        return np.mean(window)
        
    def _max_pool(self, window)
        return np.max(window)
