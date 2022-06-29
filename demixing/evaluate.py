import sys
# append script path
sys.path.append('/home/sarod/work/ttorch/smart-calibration/calibration')
from transformer_models import *
from generate_data import get_info_from_dataset

#########################################################
# Sample real observational data and make a recommendation
# Using a trained model
#########################################################

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


def evaluate_model(x):
   # Influence map size
   Ninf=64
   # extra info (separation,azimuth,elevation), log(||J||,||C||,|Inf|), LLR, log(freq)
   Nextra=8
   Ninput=Ninf*Ninf+Nextra
   # Directions (including target) == heads
   K=6
   # hidden dimension per head (must be a multiple of heads)
   Nmodel=66
   
   n_heads=K
   input_dims=Ninput*n_heads
   model_dims=Nmodel*n_heads
   
   # num_layers below indicate how many attention blocks are stacked
   net=TransformerEncoder(num_layers=1,input_dim=input_dims, model_dim=model_dims, num_heads=n_heads, num_classes=K-1, dropout=0.6).to(mydevice)
   
   checkpoint=torch.load('./net.model',map_location=mydevice)
   net.load_state_dict(checkpoint['model_state_dict'])
   net.eval()

   inputs=torch.from_numpy(x).to(mydevice)
   inputs=inputs[None,]
   
   outputs=net(inputs)
   
   print('Output:')
   print(outputs.cpu().data)


if __name__ == '__main__':
  # args : (absolute path to MS) TIME(MINUTES)
  import sys,glob
  argc=len(sys.argv)
  if argc==3:
   x=get_info_from_dataset(glob.glob(sys.argv[1]),float(sys.argv[2]),Ninf=64)
   evaluate_model(x)
  else:
   print('Usage: '+sys.argv[0]+' \'MS*pattern\' time(seconds)')

  exit()

