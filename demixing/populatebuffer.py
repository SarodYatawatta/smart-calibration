import numpy as np
import sys,os
# append script path
sys.path.append(os.path.relpath('../calibration'))
sys.path.append(os.path.relpath('../elasticnet'))

from transformer_models import *

#########################################################
# Populate trained dataset to overcome class imbalance
#########################################################

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

R=ReplayBuffer(4000,(input_dims,),(K-1,))
R.load_checkpoint()
r_filled=R.mem_cntr

print(r_filled)
yy=np.zeros(r_filled,dtype=int)
for ci in range(r_filled):
    for bit in R.y[ci]:
        yy[ci]=(yy[ci]<<1)|int(bit)

print(yy.shape)
for ci in range(32):
  z=np.where(yy==ci)
  z=z[0]
  if len(z)>0:
    print(R.y[z[0]])
  print('%d %d'%(ci,len(z)))
print(R.x[:r_filled].shape)

from imblearn.combine import SMOTETomek
smote_enn = SMOTETomek(random_state=0)
smote_enn.set_params(sampling_strategy='not minority')
params=smote_enn.get_params()
print(params)
#X_res,y_res=smote_enn.fit_resample(R.x[:r_filled],yy)
