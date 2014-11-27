from layer import Layer
import numpy as np

class PoolingLayer(Layer):
    def __init__(self, winSize, type='max'):
        self.winSize = winSize
        self.type = type
        self.poolFunc = {'max': self._max_pool, 'mean': self._mean_pool}
        
    def forward_prop(self, maps):
        # check square maps
        assert(maps.shape[1] == maps.shape[2])
        # check integer number of pooling windows
        assert(maps.shape[1]%self.winSize == 0)
        
        numMaps = maps.shape[0]
        poolW = maps.shape[1]/self.winSize
        
        pooled = np.zeros((numMaps, poolW, poolW))
        if self.type == 'max':
            pooledLoc = np.zeros((numMaps, poolW, poolW, 2))
        
        for m in xrange(self.numMaps):
            for i in xrange(poolW):
                for j in xrange(poolW):
                    pool = self.poolFunc[self.type](maps(m, i*self.winSize:(i+1)*self.winSize, j*self.winSize:(j+1)*self.winSize))
                    
                    if self.type == 'max':
                        pooled[m,i,j] = pool[0]
                        pooledLoc[m,i,j,0] = pool[1,0]+i
                        pooledLoc[m,i,j,1] = pool[1,1]+j
                    else:
                        pooled[m,i,j] = pool
        
        if self.type == 'max':          
            return (pooled, pooledLoc)
        else:
            return pooled
        
    def backward_prop(self, maps, error):
        # check same number of maps
        assert(maps.shape[0] == error.shape[0])
        # check square maps
        assert(maps.shape[1] == maps.shape[2])
        assert(error.shape[1] == error.shape[2])
        # check error size corresponds to winSize
        assert(error.shape[1] == maps.shape[1]/self.winSize)
        
        newDelta = np.zeros(maps.shape)
        
        if self.type == 'max':
            (pooled, pooledLoc) = self.forward_prop(inp)
            
            for m in xrange(pooledLoc.shape[0]):
                for i in xrange(pooledLoc.shape[1]):
                    for j in xrange(pooledLoc.shape[2]):
                        x = pooledLoc[m,i,j,0]
                        y = pooledLoc[m,i,j,1]
                        newDelta[m,x,y] = error[m,i,j]
        else:
            for m in xrange(error.shape[0]):
                for i in xrange(error.shape[1]):
                    for j in xrange(error.shape[2]):
                        newDelta[m, i*self.winsize:(i+1)*self.winSize, j*self.winSize:(j+1)*self.winSize] = error[m,i,j]*np.ones((self.winSize,self.winSize))
            
        return newDelta
        
    def _mean_pool(self, window)
        return np.mean(window)
        
    def _max_pool(self, window)
        return (np.max(window), np.unravel_index(np.argmax(window),(self.winSize,self.winSize)))
        
