import numpy as np
import cv2

class Cusenv():

  def __init__(self):
    super(Cusenv,self).__init__()
    self.img = []
    self.prev = []

  def reset(self, raw_x , raw_n):#This reset() is specific for denoising
    
      self.img = raw_x + raw_n
      self.prev = raw_x + raw_n
      self.ground_truth = raw_x

      return self.img

  def step(self, action, t): #This step() is specific for denoising
      move = action.astype(np.float32)
      move = (move[:,np.newaxis,:,:]-13.0)/255
      self.img = self.img+move
      r = 255*np.square(self.prev - self.ground_truth)-255*np.square(self.img-self.ground_truth) # Reward
      s_prime = self.img.copy()
      self.prev = self.img.copy()
      
      if t>4:
          done = True
      else:
          done = False
          
      return s_prime, r, done


