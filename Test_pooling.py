from pooling import PoolingLayer
import numpy as np
import unittest

class PoolingLayerTest(unittest.TestCase):
    def setUp(self):
        self.maps4x4 = np.array([[[4, 0, 0, 4],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [4, 0, 0, 4]],
                                  
                                 [[8, 0, 0, 8],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [8, 0, 0, 8]]])
                                  
        self.error2x2 = np.array([[[1, 2],
                                   [3, 4]],
                           
                                  [[4, 3],
                                   [2, 1]]])
        
        self.maps9x9 = np.array([[[9, 0, 0, 0, 9, 0, 0, 0, 9],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [9, 0, 0, 0, 9, 0, 0, 0, 9],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [9, 0, 0, 0, 9, 0, 0, 0, 9]],
                          
                                 [[18, 0, 0, 0, 18, 0, 0, 0, 18],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [18, 0, 0, 0, 18, 0, 0, 0, 18],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [18, 0, 0, 0, 18, 0, 0, 0, 18]]])
                          
        self.error3x3 = np.array([[[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]],
                           
                                  [[9, 8, 7],
                                   [6, 5, 4],
                                   [3, 2, 1]]])
                        
        self.max_pool4x4 = PoolingLayer(4,2,'max')
        self.mean_pool4x4 = PoolingLayer(4,2,'mean')                
                        
        self.max_pool9x9 = PoolingLayer(9,3,'max')
        self.mean_pool9x9 = PoolingLayer(9,3,'mean')
        
    def test_max_forward_prop_4x4(self):
        pool = self.max_pool4x4.forward_prop(self.maps4x4)
        self.assertTrue((pool[0] == np.array([4*np.ones((2,2)), 8*np.ones((2,2))])).all())
        
    def test_max_backward_prop_4x4(self):
        ans = np.array([[[1, 0, 0, 2],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [3, 0, 0, 4]],
                          
                        [[4, 0, 0, 3],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [2, 0, 0, 1]]])
                    
        self.max_pool4x4.forward_prop(self.maps4x4)     
        back = self.max_pool4x4.backward_prop(self.error2x2)
        self.assertTrue((back == ans).all())
        
    def test_mean_forward_prop_4x4(self):
        pool = self.mean_pool4x4.forward_prop(self.maps4x4)
        self.assertTrue((pool == np.array([np.ones((2,2)), 2*np.ones((2,2))])).all())
       
    def test_mean_backward_prop_4x4(self):
        ans = np.array([[[1., 1., 2., 2.],
                         [1., 1., 2., 2.],
                         [3., 3., 4., 4.],
                         [3., 3., 4., 4.]],
                          
                        [[4., 4., 3., 3.],
                         [4., 4., 3., 3.],
                         [2., 2., 1., 1.],
                         [2., 2., 1., 1.]]])/4
                         
        self.mean_pool4x4.forward_prop(self.maps4x4)
        back = self.mean_pool4x4.backward_prop(self.error2x2)
        self.assertTrue((back == ans).all())    
        
    def test_max_forward_prop_9x9(self):
        pool = self.max_pool9x9.forward_prop(self.maps9x9)
        self.assertTrue((pool[0] == np.array([9*np.ones((3,3)), 18*np.ones((3,3))])).all())
        
    def test_max_backward_prop_9x9(self):
        ans = np.array([[[1, 0, 0, 0, 2, 0, 0, 0, 3],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [4, 0, 0, 0, 5, 0, 0, 0, 6],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [7, 0, 0, 0, 8, 0, 0, 0, 9]],
                         
                        [[9, 0, 0, 0, 8, 0, 0, 0, 7],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [6, 0, 0, 0, 5, 0, 0, 0, 4],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 2, 0, 0, 0, 1]]])
               
        self.max_pool9x9.forward_prop(self.maps9x9)          
        back = self.max_pool9x9.backward_prop(self.error3x3)
        self.assertTrue((back == ans).all())
        
    def test_mean_forward_prop_9x9(self):
        pool = self.mean_pool9x9.forward_prop(self.maps9x9)
        self.assertTrue((pool == np.array([np.ones((3,3)), 2*np.ones((3,3))])).all())
        
    def test_mean_backward_prop_9x9(self):
        ans = np.array([[[1., 1., 1., 2., 2., 2., 3., 3., 3.],
                         [1., 1., 1., 2., 2., 2., 3., 3., 3.],
                         [1., 1., 1., 2., 2., 2., 3., 3., 3.],
                         [4., 4., 4., 5., 5., 5., 6., 6., 6.],
                         [4., 4., 4., 5., 5., 5., 6., 6., 6.],
                         [4., 4., 4., 5., 5., 5., 6., 6., 6.], 
                         [7., 7., 7., 8., 8., 8., 9., 9., 9.],
                         [7., 7., 7., 8., 8., 8., 9., 9., 9.],
                         [7., 7., 7., 8., 8., 8., 9., 9., 9.]],
                          
                        [[9., 9., 9., 8., 8., 8., 7., 7., 7.],
                         [9., 9., 9., 8., 8., 8., 7., 7., 7.],
                         [9., 9., 9., 8., 8., 8., 7., 7., 7.],
                         [6., 6., 6., 5., 5., 5., 4., 4., 4.],
                         [6., 6., 6., 5., 5., 5., 4., 4., 4.],
                         [6., 6., 6., 5., 5., 5., 4., 4., 4.], 
                         [3., 3., 3., 2., 2., 2., 1., 1., 1.],
                         [3., 3., 3., 2., 2., 2., 1., 1., 1.],
                         [3., 3., 3., 2., 2., 2., 1., 1., 1.]]])/9
                 
        self.mean_pool9x9.forward_prop(self.maps9x9)       
        back = self.mean_pool9x9.backward_prop(self.error3x3)
        self.assertTrue(np.allclose(back, ans))
    
if __name__ == "__main__":
    unittest.main()
