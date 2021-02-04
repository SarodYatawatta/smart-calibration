from calib_td3 import ReplayBuffer
import torch
import torchvision

def gray_to_rgb(x):
 dims=len(x.shape)
 if dims==3: # one tile
   tiles=1
   y=0.8*x/(x.max()-x.min())+0.5
 else: # dims==4, many tiles
   tiles=x.shape[0]
   y=torch.zeros_like(x)
   for cn in range(tiles):
     z=x[cn]
     y[cn]=0.8*z/(z.max()-z.min())+0.5

 return y

rb=ReplayBuffer(1000,[1,128,128],K=4,M=4,n_actions=4)
rb.load_checkpoint()
a=torch.Tensor(rb.state_memory_img[0:540:10])
b=gray_to_rgb(a)
torchvision.utils.save_image(b,'foo.png',scale_each=True,normalize=True)

for x in rb.reward_memory[0:540:1]:
 print(x)
