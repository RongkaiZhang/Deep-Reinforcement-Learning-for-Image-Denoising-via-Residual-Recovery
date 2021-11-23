import numpy as np
import cv2
#first_layer = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros')
#second_layer = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=True, padding_mode='zeros')
#



class Cusenv():

  def __init__(self):
    super(Cusenv,self).__init__()
    self.img = []
    self.prev = []

  def reset(self, raw_x , raw_n):
      self.img = raw_x + raw_n
      self.prev = raw_x + raw_n
      self.ground_truth = raw_x
      #self.img = np.maximum(0, self.img)
      #self.img = np.minimum(1, self.img)
      #self.prev = np.maximum(0, self.prev)
      #self.prev = np.minimum(1, self.prev)
      return self.img

  def step(self, action, t):
      move = action.astype(np.float32)
      move = (move[:,np.newaxis,:,:]-13.0)/255
      self.img = self.img+move
      #self.img = np.maximum(0, self.img)
      #self.img = np.minimum(1, self.img)
      r = 255*np.square(self.prev - self.ground_truth)-255*np.square(self.img-self.ground_truth)
      #r_rl = cv2.GaussianBlur(r[0,0],(5,5),sigmaX=0.5)
      #r_rl = r_rl[np.newaxis,np.newaxis,:,:]
      #r_rl = np.maximum(0,r)
      #r_rl = np.count_nonzero(r_rl)
      s_prime = self.img.copy()
      self.prev = self.img.copy()
      if t>4:
          done = True
      else:
          #r = r #score for surviving
          done = False
          #r = r+1.0
      #r = r-(1-r_rl/4900)
      return s_prime, r, done


