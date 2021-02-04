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

# MS name to use as filename base 'XX_SB001_MS.solutions'
# broken to 2 parts
MS1='L_'
MS2='.MS'


# stations
N=62
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
initialrho='admm_rho0.txt' # initial values for rho, determined analytically
ff=open(outskymodel,'w+')
ff1=open(outskymodel1,'w+')
gg=open(outcluster,'w+')
gg1=open(outcluster1,'w+')
skl=open(skycluster,'w+')
arh=open(initialrho,'w+')

# phase center
ra0=0
dec0=math.pi/2


# number of sources at the center, included in calibration
Kc=1
# generate random sources in [-lmin,lmax] at the phase center
lmin=0.1
l=(np.random.rand(Kc)-0.5)*lmin
m=(np.random.rand(Kc)-0.5)*lmin
n=(np.sqrt(1-np.power(l,2)-np.power(m,2))-1)
# intensities, uniform in [10,100]/10 : ~=10, so that rho matches
sI=((np.random.rand(Kc)*90)+10)/10
sI=sI/np.min(sI)*3 # min flux 3 Jy
# spectral indices
sP=np.random.randn(Kc)

#%%%%%%%%% weak sources
# weak sources in background
M=150
a=0.01
b=0.1#  flux in [0.01 0.1]
alpha=-2
nn=np.random.rand(M)
sII=np.power(np.power(a,(alpha+1))+nn*(np.power(b,(alpha+1))-np.power(a,(alpha+1))),(1/(alpha+1)))
# for a FOV 16.0x16.0,
l0=(np.random.rand(M)-0.5)*15.5*math.pi/180
m0=(np.random.rand(M)-0.5)*15.5*math.pi/180
n0=(np.sqrt(1-np.power(l0,2)-np.power(m0,2))-1)

# extended sources
# name h m s d m s I Q U V spectral_index1 spectral_index2 spectral_index3 RM extent_X(rad) extent_Y(rad) pos_angle(rad) freq0
M1=20
a=0.01
b=0.5 # flux in [0.01 0.1]
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
gg1.write('1 1')
arh.write('# format\n')
arh.write('# cluster_id hybrid admm_rho\n')
arh.write('1 1 '+str(sum(sI))+'\n') # total flux x 1
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

gg.write('\n')
gg1.write('\n')



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


ff.write('# outlier sources (reset flux during calibration)\n')
ff1.write('# outlier sources (reset flux during calibration)\n')
gg.write('# clusters for outlier sources\n')
gg1.write('# clusters for outlier sources\n')

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
 arh.write(str(cj+2)+' 1 '+str(sI[cj]/1000)+'\n') # total apparent flux x 0.1, because outlier
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

gg.write('\n')

ff.close()
ff1.close()
gg.close()
gg1.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%

# time slots of solutions, multiply with -t tslot option for full duration
Ts=6

# storage for full solutions
gs=np.zeros((K,8*N*Ts,Nf),dtype=np.float32)

# normalize freqency 
ff=(f-f0)/f0

# randomly generate initial 8*N values, for each direction, for 1st freq
for ck in range(K):
  gs[ck,0:8*N,0]=np.random.randn(8*N)
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
     timepol=beta[0]+beta[1]*np.cos(timerange*beta[2]+beta[3])
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
       if offset==0 or offset==7:
        flist[cf].write('1\n')
       else:
        flist[cf].write('0\n')

for ci in range(Nf):
  flist[ci].close()
