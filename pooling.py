from layer import Layer
import numpy as np

class PoolingLayer(Layer):
    def __init__(self, mapSize, winSize, type='max'):
        self.mapSize = mapSize
        self.winSize = winSize
        self.type = type
        self.lastOutput = None
        self.poolFunc = {'max': self._max_pool, 'mean': self._mean_pool}
        
    def forward_prop(self, maps):
        # check same map size
        assert(maps.shape[1] == self.mapSize)
        # check square maps
        assert(maps.shape[1] == maps.shape[2])
        # check integer number of pooling windows
        assert(maps.shape[1]%self.winSize == 0)
        
        numMaps = maps.shape[0]
        poolW = self.mapSize/self.winSize
        
        pooled = np.zeros((numMaps, poolW, poolW))
        if self.type == 'max':
            pooledLoc = np.zeros((numMaps, poolW, poolW, 2))
        
        for m in xrange(numMaps):
            for i in xrange(poolW):
                for j in xrange(poolW):
                    pool = self.poolFunc[self.type](maps[m, i*self.winSize:(i+1)*self.winSize, j*self.winSize:(j+1)*self.winSize])
                    
                    if self.type == 'max':
                        pooled[m,i,j] = pool[0]
                        pooledLoc[m,i,j,0] = pool[1][0]+i*self.winSize
                        pooledLoc[m,i,j,1] = pool[1][1]+j*self.winSize
                    else:
                        pooled[m,i,j] = pool
        
        if self.type == 'max':          
            self.lastOutput = (pooled, pooledLoc)
            return (pooled, pooledLoc)
        else:
            self.lastOutput = pooled
            return pooled
        
    def backward_prop(self, error):
        # check square maps
        assert(error.shape[1] == error.shape[2])
        # check error size corresponds to winSize
        assert(error.shape[1] == self.mapSize/self.winSize)
        # check have had output
        assert(self.lastOutput != None)
        
        newDelta = np.zeros((error.shape[0],self.mapSize,self.mapSize))
        
        if self.type == 'max':
            (pooled, pooledLoc) = self.lastOutput
            
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
                        newDelta[m, i*self.winSize:(i+1)*self.winSize, j*self.winSize:(j+1)*self.winSize] = error[m,i,j]*np.ones((self.winSize,self.winSize))/self.winSize**2
            
        return newDelta
        
    def _mean_pool(self, window):
        return np.mean(window)
        
    def _max_pool(self, window):
        return (np.max(window), np.unravel_index(np.argmax(window),(self.winSize,self.winSize)))
        
