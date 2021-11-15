import numpy as np
import math
from calibration_tools import *

# only simulate solution file for SAGECal and use it for prediction
# last direction has J=I for solution (so total directions is K+1)
# when more directions are in the solution file, simulation still works
# customized for online calibration and reinforcement learning 
# enable beam during simulation
write_files=1
# sources (directions) used in calibration, 
# first one for center, 2,3,.. for outlier sources
# and last one for weak sources (so minimum 2), 3 will be the weak sources
K=4 # must match what is used in cali_main.py

# enable spatial smoothness of systematic errors
spatial_term=True # if True, enable spatial smoothness (planes in l,m for each term)
spalpha=0.95 # in [0,1], ratio of spatial term  (rest will be 1-spalpha)
# enable diffuse sky model (shapelet mode files SLSI.fits.modes  SLSQ.fits.modes  SLSU.fits.modes)
diffuse_sky=True

# MS name to use as filename base 'XX_SB001_MS.solutions'
# broken to 2 parts
MS1='L_'
MS2='.MS'


# stations
N=62
#GG N=61
#GG N=62
# baselines
B=N*(N-1)/2

# no.of frequencies to predict values
Nf=8
# frequencies where values available
f=np.linspace(115,185,Nf)*1e6
# reference freq
f0=150e6
# light speed
c=2.99792458e8


#%%%%%%%%% create sky models
outskymodel='sky0.txt' # for simulation
outskymodel1='sky.txt' # for calibration
outcluster='cluster0.txt' # for simulation
outcluster1='cluster.txt' # for calibration
skycluster='skylmn.txt' # for input to DQN
bbsskymodel='sky_bbs.txt' # input to DP3
bbsparset_dem='test_demix.parset' # DP3 parset (demixing)
bbsparset_dde='test_ddecal.parset' # DP3 parset (ddelcal)
bbsparset_pred='test_predict.parset' # DP3 parset (predict)
initialrho='admm_rho0.txt' # initial values for rho, determined analytically
ff=open(outskymodel,'w+')
ff1=open(outskymodel1,'w+')
gg=open(outcluster,'w+')
gg1=open(outcluster1,'w+')
skl=open(skycluster,'w+')
arh=open(initialrho,'w+')
bbs=open(bbsskymodel,'w+')
bbsdem=open(bbsparset_dem,'w+')
bbsdde=open(bbsparset_dde,'w+')
bbspre=open(bbsparset_pred,'w+')

# phase center
ra0=0
dec0=math.pi/2

if spatial_term:
 ltot=list()
 mtot=list()

# number of sources at the center, included in calibration
Kc=40
# generate random sources in [-lmin,lmax] at the phase center
lmin=0.01
l=(np.random.rand(Kc)-0.5)*lmin
m=(np.random.rand(Kc)-0.5)*lmin
n=(np.sqrt(1-np.power(l,2)-np.power(m,2))-1)
# intensities, uniform in [10,100]/10 : ~=10, so that rho matches
sI=((np.random.rand(Kc)*90)+10)/10
sI=sI/np.min(sI)*0.03 # min flux 0.03 Jy
# spectral indices
sP=np.random.randn(Kc)

if spatial_term:
 ltot.extend(l)
 mtot.extend(m)

#%%%%%%%%% weak sources
# weak sources in background
M=350
a=0.01
b=0.5#  flux in [0.01 0.5]
alpha=-2
nn=np.random.rand(M)
sII=np.power(np.power(a,(alpha+1))+nn*(np.power(b,(alpha+1))-np.power(a,(alpha+1))),(1/(alpha+1)))
# for a FOV 16.0x16.0,
l0=(np.random.rand(M)-0.5)*15.5*math.pi/180
m0=(np.random.rand(M)-0.5)*15.5*math.pi/180
n0=(np.sqrt(1-np.power(l0,2)-np.power(m0,2))-1)

# extended sources
# name h m s d m s I Q U V spectral_index1 spectral_index2 spectral_index3 RM extent_X(rad) extent_Y(rad) pos_angle(rad) freq0
M1=120
a=0.01
b=0.5 # flux in [0.01 0.5]
alpha=-2
nn=np.random.rand(M1)
sI1=np.power(np.power(a,(alpha+1))+nn*(np.power(b,(alpha+1))-np.power(a,(alpha+1))),(1/(alpha+1)))
# for a FOV 16.0x16.0,
l1=(np.random.rand(M1)-0.5)*15.5*math.pi/180
m1=(np.random.rand(M1)-0.5)*15.5*math.pi/180
n1=(np.sqrt(1-np.power(l1,2)-np.power(m1,2))-1)
eX=(np.random.rand(M1)-0.5)*0.5*math.pi/180
eY=(np.random.rand(M1)-0.5)*0.5*math.pi/180
eP=(np.random.rand(M1)-0.5)*180*math.pi/180


# output sources for centre cluster
# format: P0 19 59 47.0 40 40 44.0 1.0 0 0 0 -1 0 0 0 0 0 0 1000000.0
gg.write('1 1')
gg1.write('-1 1') # do not subtract, so -ve cluster id
arh.write('# format\n')
arh.write('# cluster_id hybrid admm_rho\n')
arh.write('-1 1 '+str(sum(sI)*100)+'\n') # total flux x 100

# BBS sky model
bbs.write('# (Name, Type, Patch, Ra, Dec, I, Q, U, V, ReferenceFrequency=\''+str(f0)+'\', SpectralIndex=\'[]\', MajorAxis, MinorAxis, Orientation) = format\n')
# Demix parset
bbsdem.write('steps=[demix]\n'
  +'demix.type=demixer\n'
  +'demix.blrange=[60,100000]\n'
  +'demix.demixtimestep=10\n')

# DDECal parset
bbsdde.write('steps=[ddecal]\n'
  +'ddecal.type=ddecal\n'
  +'ddecal.h5parm=./solutions.h5\n'
  +'ddecal.sourcedb=./sky\n'
  +'ddecal.mode=fulljones\n'
  +'ddecal.uvlambdamin=30\n'
  +'ddecal.usebeammodel=true\n'
  +'ddecal.solint=10\n')

bbspre.write('steps=[predict]\n'
  +'predict.type=h5parmpredict\n'
  +'predict.sourcedb=./sky\n'
  +'predict.usebeammodel=true\n'
  +'predict.onebeamperpatch=false\n'
  +'predict.applycal.correction=fulljones\n'
  +'predict.applycal.parmdb=./solutions.h5\n'
  +'predict.applycal.steps=[applycal_amp, applycal_phase]\n'
  +'predict.applycal.applycal_amp.correction=amplitude000\n'
  +'predict.applycal.applycal_phase.correction=phase000\n'
  +'predict.operation=subtract\n')

hh,mm,ss=radToRA(ra0)
dd,dmm,dss=radToDec(dec0)
centername="CENTER"
bbsdem.write('demix.targetsource=\"'+centername+'\"\n')
bbs.write(', ,'+centername+','+str(hh)+':'+str(mm)+':'+str(int(ss))+','
  +str(dd)+'.'+str(dmm)+'.'+str(int(dss))+'\n')
for cj in range(Kc):
 ra,dec=lmtoradec(l[cj],m[cj],ra0,dec0)
 hh,mm,ss=radToRA(ra)
 dd,dmm,dss=radToDec(dec)
 sname='PC'+str(cj)
 ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sI[cj])+' 0 0 0 '+str(sP[cj])+' 0 0 0 0 0 0 '+str(f0)+'\n')
 ff1.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sI[cj])+' 0 0 0 '+str(sP[cj])+' 0 0 0 0 0 0 '+str(f0)+'\n')
 gg.write(' '+sname)
 gg1.write(' '+sname)
 # output for DQN : formate cluster_id, l, m, sI, sP
 skl.write('1 '+str(l[cj])+' '+str(m[cj])+' '+str(sI[cj])+' '+str(sP[cj])+'\n')
 # BBS
 #, , CygA, 19:59:26, +40.44.00
 #CygA_4_2, POINT,    CygA, 19:59:30.433, +40.43.56.221, 4.827e+03, 0.0, 0.0, 0.0, 7.38000e+07, [-0.8], 7.63889e-03, 6.94444e-03, 1.00637e+02
 bbs.write(sname+',POINT,'+centername+','+str(hh)+':'+str(mm)+':'+str(int(ss))+','
        +str(dd)+'.'+str(dmm)+'.'+str(int(dss))+','
        +str(sI[cj])+', 0, 0, 0,'+str(f0)+',['+str(sP[cj])+'], 0, 0, 0'+'\n')

gg.write('\n')
gg1.write('\n')



bbsdem.write('demix.subtractsources=[')
bbsdde.write('ddecal.directions=[')
bbspre.write('predict.directions=[')
# output sources for outlier clusters (one source per cluster)
Kc=K-1
# generate random sources in [-lmin,lmax] at the phase center
lmin=0.7
l=(np.random.rand(Kc)-0.5)*lmin;
m=(np.random.rand(Kc)-0.5)*lmin;
n=(np.sqrt(1-np.power(l,2)-np.power(m,2))-1)
# intensities, uniform in [100,1000]
sI=((np.random.rand(Kc)*900)+100)/10
sI=sI/np.min(sI)*250 # min flux 150 Jy (will be attenuated by the beam)
# spectral indices
sP=np.random.randn(Kc)

if spatial_term:
 ltot.extend(l)
 mtot.extend(m)

ff.write('# outlier sources (reset flux during calibration)\n')
ff1.write('# outlier sources (reset flux during calibration)\n')
gg.write('# clusters for outlier sources\n')
gg1.write('# clusters for outlier sources\n')

firstcomma=False
for cj in range(Kc):
 ra,dec=lmtoradec(l[cj],m[cj],ra0,dec0)
 hh,mm,ss=radToRA(ra)
 dd,dmm,dss=radToDec(dec)
 sname='PO'+str(cj)
 ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sI[cj])+' 0 0 0 '+str(sP[cj])+' 0 0 0 0 0 0 '+str(f0)+'\n')
 # divide fluxes during calibration because of the beam
 ff1.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sI[cj]/100)+' 0 0 0 '+str(sP[cj])+' 0 0 0 0 0 0 '+str(f0)+'\n')
 gg.write(str(cj+2)+' 1 '+sname+'\n')
 gg1.write(str(cj+2)+' 1 '+sname+'\n')
 skl.write(str(cj+2)+' '+str(l[cj])+' '+str(m[cj])+' '+str(sI[cj]/100)+' '+str(sP[cj])+'\n')
 arh.write(str(cj+2)+' 1 '+str(sI[cj]/1000*100)+'\n') # total apparent flux x 0.1, because outlier
 bbs.write(', ,'+sname+','+str(hh)+':'+str(mm)+':'+str(int(ss))+','
         +str(dd)+'.'+str(dmm)+'.'+str(int(dss))+'\n')
 bbs.write(sname+'_1,POINT,'+sname+','+str(hh)+':'+str(mm)+':'+str(int(ss))+','
        +str(dd)+'.'+str(dmm)+'.'+str(int(dss))+','
        +str(sI[cj]/1000*100)+', 0, 0, 0,'+str(f0)+',['+str(sP[cj])+'], 0, 0, 0'+'\n')
 if not firstcomma:
   firstcomma=True
 else:
   bbsdem.write(',')
   bbsdde.write(',')
   bbspre.write(',')
 bbsdem.write('\"'+sname+'\"')
 bbsdde.write('\"'+sname+'\"')
 bbspre.write('['+sname+']')


bbsdem.write(']\n')
bbsdde.write(',\"'+centername+'\"]\n')
bbspre.write(']\n')
bbs.close()
bbsdem.close()
bbsdde.close()
bbspre.close()

skl.close()
arh.close()
# weak sources are grouped into one cluster
ff.write('# weak sources\n')
gg.write('# cluster for weak sources\n')
gg.write(str(K+1)+' 1 ')
for cj in range(M):
 ra,dec=lmtoradec(l0[cj],m0[cj],ra0,dec0)
 hh,mm,ss=radToRA(ra)
 dd,dmm,dss=radToDec(dec)
 sname='PW'+str(cj)
 ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sII[cj])+' 0 0 0 0 0 0 0 0 0 0 '+str(f0)+'\n')
 gg.write(str(sname)+' ')

# Gaussians
for cj in range(M1):
 ra,dec=lmtoradec(l1[cj],m1[cj],ra0,dec0)
 hh,mm,ss=radToRA(ra)
 dd,dmm,dss=radToDec(dec)
 sname='GW'+str(cj)
 ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sI1[cj])+' 0 0 0 0 0 0 0 '+str(eX[cj])+' '+str(eY[cj])+' '+str(eP[cj])+' '+str(f0)+'\n')
 gg.write(str(sname)+' ')

if diffuse_sky:
  # shapelet models
  hh,mm,ss=radToRA(ra0)
  dd,dmm,dss=radToDec(dec0)
  sname='SLSI'
  ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' 25.0 0 0 0 -0.100000 0.000000 0.000000 0.0 1.0 1.0 0.0 '+str(f0)+'\n')
  gg.write(str(sname)+' ')
  sname='SLSQ'
  ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' 0.0 25.0 0 0 -0.100000 0.000000 0.000000 0.0 1.0 1.0 0.0 '+str(f0)+'\n')
  gg.write(str(sname)+' ')
  sname='SLSU'
  ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' 0.0 0.0 25.0 0 -0.100000 0.000000 0.000000 0.0 1.0 1.0 0.0 '+str(f0)+'\n')
  gg.write(str(sname)+' ')

gg.write('\n')

ff.close()
ff1.close()
gg.close()
gg1.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%

# time slots of solutions, multiply with -t tslot option for full duration
Ts=6
#GG Ts=187
#Ts=486

# storage for full solutions
gs=np.zeros((K,8*N*Ts,Nf),dtype=np.float32)

# normalize freqency 
ff=(f-f0)/f0

if spatial_term:
  # spatial term (a0 l + a1 m + a2) planes in l,m
  a0=np.random.randn(8*N)
  a1=np.random.randn(8*N)
  a2=np.random.randn(8*N)
  a0=a0/np.linalg.norm(a0)
  a1=a1/np.linalg.norm(a1)
  a2=a2/np.linalg.norm(a2)

for ck in range(K):
  if not spatial_term:
    # randomly generate initial 8*N values, for each direction, for 1st freq
    gs[ck,0:8*N,0]=np.random.randn(8*N)
  else:
    randpart=np.random.randn(8*N)
    gs[ck,0:8*N,0]=(1-spalpha)*randpart/np.linalg.norm(randpart) + spalpha*(a0*ltot[ck]+a1*mtot[ck]+a2)
    gs[ck,0:8*N,0] /= np.linalg.norm(gs[ck,0:8*N,0])
  # also add 1 to J_00 and J_22 (real part) : every 0 and 6 value
  gs[ck,0:8*N:8] +=1.
  gs[ck,6:8*N:8] +=1.

  # generate a random polynomial over freq
  for ci in range(8*N):
    alpha=gs[ck,ci,0]
    beta=np.random.randn(3) 
    # output=alpha*(b0+b1*f+b2*f^2)
    freqpol=alpha*(beta[0]+beta[1]*ff+beta[2]*np.power(ff,2))
    gs[ck,ci]=freqpol
# now the 1-st timeslot solutions for all freqs are generated
# copy this to other timeslots
for ck in range(K):
  for ct in range(1,Ts):
    gs[ck,ct*8*N:(ct+1)*8*N]=gs[ck,0:8*N]

# now iterate over 8N for all timeslots using similar polynomials
timerange=np.arange(0,Ts)/Ts
for ck in range(K):
  for cn in range(8*N):
     beta=np.random.randn(4)
     beta=beta/np.linalg.norm(beta)
     # add DC term to time poly
     timepol=1+beta[0]+beta[1]*np.cos(timerange*beta[2]+beta[3])
     for cf in range(Nf):
       gs[ck,cn:8*N*Ts:8*N,cf] *=timepol



# open all files
flist={}
for ci in range(Nf):
  flist[ci]=open(MS1+'SB'+str(ci+1)+MS2+'.S.solutions','w+')

  flist[ci].write('#solution file created by simulate.py for SAGECal\n')
  flist[ci].write('#freq(MHz) bandwidth(MHz) time_interval(min) stations clusters effective_clusters\n')
  flist[ci].write(str(f[ci]/1e6)+' 0.183105 20.027802 '+str(N)+' '+str(K+1)+' '+str(K+1)+'\n')
  

for ct in range(Ts):
  for ci in range(8*N):
    stat=ci//8
    offset=ci-8*stat
    for cf in range(Nf):
       flist[cf].write(str(ci)+' ')
       for ck in range(K):
        flist[cf].write(str(gs[ck,ct*8*N+ci,cf])+' ')
       # last column, 1 at 0 and 6, else 0
       if offset==0 or offset==6:
        flist[cf].write('1\n')
       else:
        flist[cf].write('0\n')

for ci in range(Nf):
  flist[ci].close()
