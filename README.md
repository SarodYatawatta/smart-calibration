# Smart Calibration
Using reinforcement learning for hyperparameter tuning in calibration of radio telescopes, and in other data processing pipelines (like elastic net regerssion).

RL agent code is based on [this code](https://github.com/philtabor/Youtube-Code-Repository.git).

Implemented in PyTorch, using openai.gym.
## Elastic net regression

Files included are:

``` autograd_tools.py ```: utilities to calculate Jacobian, inverse Hessian-vector product etc.

``` enetenv.py ```: openai.gym environment

``` enet_td3.py  ```:  TD3 training

``` enet_ddpg.py ```: DDPG training

``` enet_eval.py ```: evaluation

``` lbfgsnew.py ```: LBFGS optimizer

``` main_ddpg.py ```: run this for DDPG

``` main_td3.py ```: run this for TD3

## Calibration

Additional packages: casacore-python, astropy, pyfits, openmpi

Calibration software: [SAGECal](https://github.com/nlesc-dirac/sagecal)

Imaging software: [Excon](https://sourceforge.net/projects/exconimager/)

Files included are:

``` calibration_tools.py ```: utility routines

``` simulate.py ```: simulate data

``` calibenv.py ```: openai.gym environment

``` analysis.py ```: calculate influence function/map

``` calib_td3.py ```: TD3 training

``` calib_ddpg.py ```: DDPG training

``` lbfgsnew.py ```: LBFGS optimizer

``` main_ddpg.py ```: run this for DDPG

``` main_td3.py ```: run this for TD3

``` docal.sh ```: shell wrapper to run calibration

``` doinfluence.sh ```: shell wrapper to run influence mapping

``` dosimul.sh  ```: shell wrapper to run simulations

Others:
addcol.py
changefreq.py
readcorr.py
writecorr.py
addnoise.py 
doall.sh           
calmean.sh              
