from convolution_full import ConvolutionLayer
import numpy as np
import unittest

class ConvolutionLayerTest(unittest.TestCase):
    def setUp(self):
        self.t_input_1ch = np.array([[[1, 2, 3, 4],
                                      [5, 6, 7, 8],
                                      [9, 10, 11, 12],
                                      [13, 14, 15, 16]]])
                                    
        self.t_input_2ch = np.array([[[1, 2, 3, 4],
                                      [5, 6, 7, 8],
                                      [9, 10, 11, 12],
                                      [13, 14, 15, 16]],
                                    
                                     [[17, 18, 19, 20],
                                      [21, 22, 23, 24],
                                      [25, 26, 27, 28],
                                      [29, 30, 31, 32]]])
                                    
        self.t_output_4filt = np.array([[1, 2, 3, 4],
                                        [5, 6, 7, 8],
                                        [9, 10, 11, 12],
                                        [13, 14, 15, 16]])
                                      
        self.img9x9_1ch = np.array([[[1,  1,  1,  1,  1,  1,  1,  1,  1],
                                     [1, -8,  1, -8,  1, -8,  1, -8,  1],
                                     [1,  1,  1,  1,  1,  1,  1,  1,  1],
                                     [1, -8,  1, -8,  1, -8,  1, -8,  1],
                                     [1,  1,  1,  1,  1,  1,  1,  1,  1],
                                     [1, -8,  1, -8,  1, -8,  1, -8,  1], 
                                     [1,  1,  1,  1,  1,  1,  1,  1,  1],
                                     [1, -8,  1, -8,  1, -8,  1, -8,  1],
                                     [1,  1,  1,  1,  1,  1,  1,  1,  1]]]) 
                                     
        self.img9x9_3ch = np.array([[[1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1]],
                                     
                                    [[1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1, -26,   1, -26,   1, -26,   1, -26,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1, -26,   1, -26,   1, -26,   1, -26,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1, -26,   1, -26,   1, -26,   1, -26,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1, -26,   1, -26,   1, -26,   1, -26,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1]],
                                     
                                    [[1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1],
                                     [1,   1,   1,   1,   1,   1,   1,   1,   1]]])                             
                                      
        self.filters2_1ch = np.array([[[[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]]],
                                      
                                      [[[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]]]])
                                       
        self.filters2_bias_1ch = np.array([0, 0])
        
        self.filters2_3ch = np.array([[[[0, 0, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]],
                                        
                                       [[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]],
                                        
                                       [[0, 0, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]]],
                                      
                                      [[[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]],
                                        
                                       [[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]],
                                       
                                       [[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]]]])
                                        
        self.filters2_bias_3ch = np.array([0, 0])
                                   
        self.conv_1ch_4filt = ConvolutionLayer(1,4,2)
        self.conv_2ch_1filt = ConvolutionLayer(2,1,2)
        self.conv_1ch_2filt = ConvolutionLayer(1,2,3)
        self.conv_3ch_2filt = ConvolutionLayer(3,2,3)
        
    def test_input_transform_1ch(self):
        ans = np.array([[1, 2, 5, 6, 1],
                        [2, 3, 6, 7, 1],
                        [3, 4, 7, 8, 1],
                        [5, 6, 9, 10, 1],
                        [6, 7, 10, 11, 1],
                        [7, 8, 11, 12, 1],
                        [9, 10, 13, 14, 1],
                        [10, 11, 14, 15, 1],
                        [11, 12, 15, 16, 1]])
                        
        trans = self.conv_1ch_4filt._transformInput(self.t_input_1ch)
        self.assertTrue(np.allclose(trans, ans))
        
    def test_input_transform_2ch(self):
        ans = np.array([[1, 2, 5, 6, 17, 18, 21, 22, 1],
                        [2, 3, 6, 7, 18, 19, 22, 23, 1],
                        [3, 4, 7, 8, 19, 20, 23, 24, 1],
                        [5, 6, 9, 10, 21, 22, 25, 26, 1],
                        [6, 7, 10, 11, 22, 23, 26, 27, 1],
                        [7, 8, 11, 12, 23, 24, 27, 28, 1],
                        [9, 10, 13, 14, 25, 26, 29, 30, 1],
                        [10, 11, 14, 15, 26, 27, 30, 31, 1],
                        [11, 12, 15, 16, 27, 28, 31, 32, 1]])
                        
        trans = self.conv_2ch_1filt._transformInput(self.t_input_2ch)
        self.assertTrue(np.allclose(trans, ans))
        
    def test_output_transform(self):
        ans = np.array([[[1, 5],
                        [9, 13]],
        
                        [[2, 6],
                         [10, 14]],
                         
                        [[3, 7],
                         [11, 15]],
                         
                        [[4, 8],
                         [12, 16]]])
                         
        trans = self.conv_1ch_4filt._transformOutput(self.t_output_4filt)
        self.assertTrue(np.allclose(trans, ans))
        
    def test_init_weights(self):
        ans = np.array([[0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [1, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 0]])
        
        self.conv_1ch_2filt.init_weights(self.filters2_1ch, self.filters2_bias_1ch)
        self.assertTrue(np.allclose(self.conv_1ch_2filt.fullLayer.W, ans))
        
    def test_fprop_2filt_1ch(self):
        ans = np.tanh(np.array([[[-8,  1, -8,  1, -8,  1, -8],
                                 [ 1,  1,  1,  1,  1,  1,  1],
                                 [-8,  1, -8,  1, -8,  1, -8],
                                 [ 1,  1,  1,  1,  1,  1,  1],
                                 [-8,  1, -8,  1, -8,  1, -8],
                                 [ 1,  1,  1,  1,  1,  1,  1],
                                 [-8,  1, -8,  1, -8,  1, -8]],
                         
                                [[ 0,  -9,  0,  -9,  0,  -9,  0],
                                 [-9, -27, -9, -27, -9, -27, -9],
                                 [ 0,  -9,  0,  -9,  0,  -9,  0],
                                 [-9, -27, -9, -27, -9, -27, -9],
                                 [ 0,  -9,  0,  -9,  0,  -9,  0],
                                 [-9, -27, -9, -27, -9, -27, -9],
                                 [ 0,  -9,  0,  -9,  0,  -9,  0]]]))
        
        self.conv_1ch_2filt.init_weights(self.filters2_1ch, self.filters2_bias_1ch)
        conv = self.conv_1ch_2filt.forward_prop(self.img9x9_1ch)
        self.assertTrue(np.allclose(conv, ans))
        
    def test_fprop_2filt_3ch(self):
        ans = np.tanh(np.array([[[-26,   1, -26,   1, -26,   1, -26],
                                 [  1,   1,   1,   1,   1,   1,   1],
                                 [-26,   1, -26,   1, -26,   1, -26],
                                 [  1,   1,   1,   1,   1,   1,   1],
                                 [-26,   1, -26,   1, -26,   1, -26],
                                 [  1,   1,   1,   1,   1,   1,   1],
                                 [-26,   1, -26,   1, -26,   1, -26]],
                         
                                [[  0, -27,   0, -27,   0, -27,   0],
                                 [-27, -81, -27, -81, -27, -81, -27],
                                 [  0, -27,   0, -27,   0, -27,   0],
                                 [-27, -81, -27, -81, -27, -81, -27],
                                 [  0, -27,   0, -27,   0, -27,   0],
                                 [-27, -81, -27, -81, -27, -81, -27],
                                 [  0, -27,   0, -27,   0, -27,   0]]]))
        
        self.conv_3ch_2filt.init_weights(self.filters2_3ch, self.filters2_bias_3ch)
        conv = self.conv_3ch_2filt.forward_prop(self.img9x9_3ch)
        self.assertTrue(np.allclose(conv, ans))
        
if __name__ == "__main__":
    unittest.main()
    
