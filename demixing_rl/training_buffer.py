import numpy as np
import pickle


class TrainingBuffer(object):
    def __init__(self, max_size, n_input=8, n_output=6):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.n_x=n_input
        self.n_y=n_output

        # X: input
        self.x_= np.zeros((self.mem_size, self.n_x), dtype=np.float32)
        # Y: output
        self.y_= np.zeros((self.mem_size, self.n_y), dtype=np.float32)

        self.filename='databuffer.npy' # for saving object

    def store_observation(self, x, y):
        index = self.mem_cntr % self.mem_size
        self.x_[index] = x
        self.y_[index] = y
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # return x,y as separate numpy arrays
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        x= self.x_[batch]
        y= self.y_[batch]

        return x,y

    def save_checkpoint(self):
        with open(self.filename,'wb') as f:
          pickle.dump(self,f)
        
    def load_checkpoint(self):
        with open(self.filename,'rb') as f:
          temp=pickle.load(f)
          self.n_x=temp.n_x
          self.n_y=temp.n_y
          self.mem_size=temp.mem_size
          self.mem_cntr=temp.mem_cntr
          self.x_=temp.x_
          self.y_=temp.y_

    def reset(self):
        self.mem_cntr=0
