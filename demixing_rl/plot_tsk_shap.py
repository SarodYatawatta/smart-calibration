import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.callbacks import EarlyStoppingACC
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK
import matplotlib.pyplot as plt
import shap

# directions, including target
K=6
# influence map size (not needed)
Ninf=128
# metadata size
M=3*K+2
# number of samples to collect (buffer size)
n_samples=1
# cadence to save buffer
n_cadence=2

# Define TSK model parameters
n_rule = 3  # Num. of rules per each input
lr = 0.01  # learning rate
consbn = True
order = 1
n_class=K-1

gmf = nn.Sequential(
        AntecedentGMF(in_dim=M, n_rule=n_rule, high_dim=True, init_center=None),
        nn.LayerNorm(n_rule),
        nn.ReLU()
    )# set high_dim=True is highly recommended.

# --------- Define full TSK model ------------
model = nn.Sequential(
        TSK(in_dim=M, out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=None),
        nn.Tanh()
        )

model.load_state_dict(torch.load('tsk.model'))


from training_buffer import TrainingBuffer


# we do not include target as an output, n_output=K-1
buffer=TrainingBuffer(n_samples,n_input=M,n_output=K-1)
buffer.load_checkpoint()

X=buffer.x_[:buffer.mem_cntr]
y=buffer.y_[:buffer.mem_cntr]

META_SCALE=1e3
xmean=np.zeros(M,dtype=np.float32)
xmean[0:5]=64
xmean[12:18]=30
xmean[18]=20
xmean[19]=50
xmean/=META_SCALE
X=X-xmean

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
explainer=shap.GradientExplainer(model,torch.tensor(x_train))
shap_values=explainer(torch.tensor(x_test))

# indices: 0: CasA, 1: CygA, 2: HerA, 3: VirA, 4: TauA
name=['CasA','CygA','HerA','VirA','TauA']
for idx in range(5):
   shap_slice=shap_values[:,:,idx]
   shap_slice.feature_names=['CasA Sep','CasA Az', 'CasA El',\
      'CygA Sep','CygA Az', 'CygA El',\
      'HerA Sep','HerA Az', 'HerA El',\
      'TauA Sep','TauA Az', 'TauA El',\
      'VirA Sep','VirA Az', 'VirA El',\
      'Target Sep','Target Az', 'Target El',\
      'Freq', 'Stations']
   shap.plots.beeswarm(shap_slice,show=False,color=plt.get_cmap('cool'))
   plt.savefig('shap_'+str(name[idx])+'.png',dpi=300,bbox_inches='tight')
   plt.clf()
