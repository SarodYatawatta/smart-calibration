import math
import os
import time
import numpy as np
import casacore.tables as ctab
from casacore.measures import measures
from casacore.quanta import quantity
from calibration_tools import *
from influence_tools import analysis_uvw_perdir,calculate_separation,get_cluster_centers

# executables
makems_binary='/home/sarod/scratch/software/bin/makems'
sagecal='/home/sarod/work/DIRAC/sagecal/build/dist/bin/sagecal_gpu'
sagecal_mpi='/home/sarod/work/DIRAC/sagecal/build/dist/bin/sagecal-mpi_gpu'
excon='/home/sarod/work/excon/src/MS/excon'

# number of frequencies
Nf=3
# HBA or LBA ?
hba=(np.random.choice([0,1])==1)

# epoch coordinate UTC 
mydm=measures()
x='3826896.235129999928176m'
y='460979.4546659999759868m'
z='5064658.20299999974668m'
mypos=mydm.position('ITRF',x,y,z)
t0=time.mktime(time.gmtime())
mytime=mydm.epoch('UTC',str(t0)+'s')

# Full time duration (slots), multiply with -t Tdelta option for full duration
Ts=2
Tdelta=10
# integration time (s)
Tint=1

valid_field=False
# loop till we find a valid direction (above horizon)
while not valid_field:
  # field coords (rad)
  ra0=np.random.rand(1)*math.pi*2
  dec0=np.random.rand(1)*math.pi/2
  ra0=ra0[0]
  dec0=dec0[0]
  myra=quantity(str(ra0)+'rad')
  mydec=quantity(str(dec0)+'rad')
  mydir=mydm.direction('J2000',myra,mydec)
  mydm.doframe(mytime)
  mydm.doframe(mypos)
  riseset=mydm.rise(mydir)
  rt_avg=quantity(0,'deg')
  if riseset['rise']=='above' and riseset['set']=='above':
      valid_field=True
  elif riseset['rise']=='below' or riseset['set']=='below':
      valid_field=False 
  else:
      # find the highest 
      rt_avg=0.5*(riseset['rise']+riseset['set'])
      valid_field=True

# convert rt_avg to seconds
rt_sec=(rt_avg.totime().get_value())*24*3600
t0 = t0 + rt_sec
strtime=time.strftime('%Y/%m/%d/%H:%M:%S',time.gmtime(t0))


hh,mm,ss=radToRA(ra0)
dd,dmm,dss=radToDec(dec0)

if hba:
  atable='HBA/ANTENNA'
else:
  atable='LBA/ANTENNA'

# get antennas
tt=ctab.table(atable,readonly=True)
N=tt.nrows()
tt.close()
# baselines
B=N*(N-1)//2

# generate makems config
# need to have both makems.parset and makems.cfg present
makems_parset='makems.parset'
msout='test.MS'
ff=open(makems_parset,'w+')
ff.write('NParts=1\n'
  +'NBands=1\n'
  +'NFrequencies=1\n'
  +'StartFreq=150e6\n' # this will be reset later
  +'StepFreq=180e3\n'
  +'StartTime='+strtime+'\n'
  +'StepTime='+str(Tint)+'\n'
  +'NTimes='+str(Ts*Tdelta)+'\n'
  +'RightAscension='+str(hh)+':'+str(mm)+':'+str(int(ss))+'\n'
  +'Declination='+str(dd)+'.'+str(dmm)+'.'+str(int(dss))+'\n'
  +'WriteAutoCorr=T\n'
  +'AntennaTableName=./'+str(atable)+'\n'
  +'MSName='+str(msout)+'\n'
)
ff.close()
os.system('cp '+makems_parset+' makems.cfg')
os.system(makems_binary)

# output will be msout_p0
msoutp0=msout+'_p0'

os.system('rsync -a ./LBA/FIELD '+msoutp0+'/')
# update FIELD table
field=ctab.table(msoutp0+'/FIELD',readonly=False)
delay_dir=field.getcol('DELAY_DIR')
phase_dir=field.getcol('PHASE_DIR')
ref_dir=field.getcol('REFERENCE_DIR')
lof_dir=field.getcol('LOFAR_TILE_BEAM_DIR')

ci=0
delay_dir[ci][0][0]=ra0
delay_dir[ci][0][1]=dec0
phase_dir[ci][0][0]=ra0
phase_dir[ci][0][1]=dec0
ref_dir[ci][0][0]=ra0
ref_dir[ci][0][1]=dec0
lof_dir[ci][0]=ra0
lof_dir[ci][1]=dec0

field.putcol('DELAY_DIR',delay_dir)
field.putcol('PHASE_DIR',phase_dir)
field.putcol('REFERENCE_DIR',ref_dir)
field.putcol('LOFAR_TILE_BEAM_DIR',lof_dir)
field.close()

if hba:
  os.system('rsync -a ./HBA/LOFAR_ANTENNA_FIELD '+msoutp0+'/')
else:
  os.system('rsync -a ./LBA/LOFAR_ANTENNA_FIELD '+msoutp0+'/')

# remove old files
os.system('rm -rf L_SB*.MS L_SB*fits')
# frequencies
if hba:
    freqlist=np.linspace(110,180,num=Nf)*1e6
else:
    freqlist=np.linspace(30,70,num=Nf)*1e6

f0=np.mean(freqlist)

for ci in range(Nf):
    MS='L_SB'+str(ci)+'.MS'
    os.system('rsync -a '+msoutp0+'/ '+MS)
    os.system('python changefreq.py '+MS+' '+str(freqlist[ci]))

#state - ra,dec,flux,freq

#########################################################################
# sky model/error simulation

# simulate target field and outlier, the remaining 4 clusters are part of A-team
# Sources (directions) used in calibration: 
# first one for center, 1,2,3,.. for outlier sources
# and last one for weak sources (so minimum 2), 3 will be the weak sources
K=6 # total must match = (A-team clusters + 1)

# weak sources in background
# point
M=350
# extended
M1=120
# number of sources at the center, included in calibration
Kc=40


outskymodel='sky0.txt' # for simulation
outskymodel1='sky.txt' # for calibration
outcluster='cluster0.txt' # for simulation
outcluster1='cluster.txt' # for calibration
initialrho='admm_rho.txt' # values for rho, determined analytically
ff=open(outskymodel,'w+')
ff1=open(outskymodel1,'w+')
gg=open(outcluster,'w+')
gg1=open(outcluster1,'w+')
arh=open(initialrho,'w+')


# generate random sources in [-lmin,lmin] at the phase center
lmin=0.2
l=(np.random.rand(Kc)-0.5)*lmin
m=(np.random.rand(Kc)-0.5)*lmin
n=(np.sqrt(1-np.power(l,2)-np.power(m,2))-1)
# intensities, power law, exponent -2 
alpha=-2.0
a=0.1
b=200.0#  flux in [0.1 200]
sIuniform=np.random.rand(Kc)
sI=np.power(np.power(a,(alpha+1))+sIuniform*(np.power(b,(alpha+1))-np.power(a,(alpha+1))),(1/(alpha+1)))
# spectral indices
sP=np.random.randn(Kc)

#%%%%%%%%% weak sources
a=0.01
b=0.5#  flux in [0.01 0.5]
alpha=-2.0
nn=np.random.rand(M)
sII=np.power(np.power(a,(alpha+1))+nn*(np.power(b,(alpha+1))-np.power(a,(alpha+1))),(1/(alpha+1)))
# for a FOV 30.0x30.0,
l0=(np.random.rand(M)-0.5)*25.5*math.pi/180
m0=(np.random.rand(M)-0.5)*25.5*math.pi/180
n0=(np.sqrt(1-np.power(l0,2)-np.power(m0,2))-1)

# extended sources
# name h m s d m s I Q U V spectral_index1 spectral_index2 spectral_index3 RM extent_X(rad) extent_Y(rad) pos_angle(rad) freq0
a=0.01
b=0.5 # flux in [0.01 0.5]
alpha=-2.0
nn=np.random.rand(M1)
sI1=np.power(np.power(a,(alpha+1))+nn*(np.power(b,(alpha+1))-np.power(a,(alpha+1))),(1/(alpha+1)))
# for a FOV 30.0x30.0,
l1=(np.random.rand(M1)-0.5)*25.5*math.pi/180
m1=(np.random.rand(M1)-0.5)*25.5*math.pi/180
n1=(np.sqrt(1-np.power(l1,2)-np.power(m1,2))-1)
eX=(np.random.rand(M1)-0.5)*0.5*math.pi/180
eY=(np.random.rand(M1)-0.5)*0.5*math.pi/180
eP=(np.random.rand(M1)-0.5)*180*math.pi/180


# output sources for centre cluster
# format: P0 19 59 47.0 40 40 44.0 1.0 0 0 0 -1 0 0 0 0 0 0 1000000.0
gg.write('1 1')
gg1.write('1 1') # do subtract target as well
arh.write('# format\n')
arh.write('# cluster_id hybrid admm_rho\n')
arh.write('1 1 '+str(sum(sI)*10/Kc)+'\n')
arh.close()

for cj in range(Kc):
 ra,dec=lmtoradec(l[cj],m[cj],ra0,dec0)
 hh,mm,ss=radToRA(ra)
 dd,dmm,dss=radToDec(dec)
 sname='PC'+str(cj)
 ff.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sI[cj])+' 0 0 0 '+str(sP[cj])+' 0 0 0 0 0 0 '+str(f0)+'\n')
 ff1.write(sname+' '+str(hh)+' '+str(mm)+' '+str(int(ss))+' '+str(dd)+' '+str(dmm)+' '+str(int(dss))+' '+str(sI[cj])+' 0 0 0 '+str(sP[cj])+' 0 0 0 0 0 0 '+str(f0)+'\n')
 gg.write(' '+sname)
 gg1.write(' '+sname)


gg.write('\n')
gg1.write('\n')
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

# python ./convertmodel.py ../A-Team_lowres.skymodel base.sky base.cluster base.rho
# python ./convertmodel.py ../A-Team_lowres-update.skymodel base.sky base.cluster base.rho
os.system('cp '+outskymodel+' tmp.sky')
os.system('cat base.sky > '+outskymodel)
os.system('cat tmp.sky >> '+outskymodel)
os.system('cp '+outskymodel1+' tmp.sky')
os.system('cat base.sky > '+outskymodel1)
os.system('cat tmp.sky >> '+outskymodel1)
os.system('cp '+outcluster+' tmp.cluster')
os.system('cat base.cluster > '+outcluster)
os.system('cat tmp.cluster >> '+outcluster)
os.system('cp '+outcluster1+' tmp.cluster')
os.system('cat base.cluster > '+outcluster1)
os.system('cat tmp.cluster >> '+outcluster1)
os.system('cp '+initialrho+' tmp.rho')
os.system('cat base.rho > '+initialrho)
os.system('cat tmp.rho >> '+initialrho)

# get separation of each cluster from target,
# negative separation given if a cluster is below horizon
separation,azimuth,elevation=calculate_separation(outskymodel1,outcluster1,ra0,dec0,mydm)

#########################################################################
# simulate errors for K directions, attenuate those errors
# target = column K-1
# outliser = columns 0..K-2

# storage for full solutions
gs=np.zeros((K,8*N*Ts,Nf),dtype=np.float32)

# normalized freqency
norm_f=(freqlist-f0)/f0

for ck in range(K):
  # attenuate random seed
  gs[ck,0:8*N,0]=np.random.randn(8*N)*0.01
  # also add 1 to J_00 and J_22 (real part) : every 0 and 6 value
  gs[ck,0:8*N:8] +=1.
  gs[ck,6:8*N:8] +=1.

  # generate a random polynomial over freq
  for ci in range(8*N):
    alpha=gs[ck,ci,0]
    beta=np.random.randn(3)
    # output=alpha*(b0+b1*f+b2*f^2)
    freqpol=alpha*(beta[0]+beta[1]*norm_f+beta[2]*np.power(norm_f,2))
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


# For randomly excluding some of the A-team sources,
# attenuate randomly selected clusters from the A-team (columns 0...K-2)
# how many to exclude ?
Kcut=np.random.choice(np.arange(K-1))
# select this many for attenuation
Kexclude=np.random.choice(K-1,Kcut,replace=False)
if Kcut>0:
  for ck in Kexclude:
     gs[ck] *=1e-12

# open all files
flist={}
for ci in range(Nf):
  MS='L_SB'+str(ci)+'.MS'
  flist[ci]=open(MS+'.S.solutions','w+')

  flist[ci].write('#solution file created by simulate.py for SAGECal\n')
  flist[ci].write('#freq(MHz) bandwidth(MHz) time_interval(min) stations clusters effective_clusters\n')
  flist[ci].write(str(freqlist[ci]/1e6)+' 0.183105 20.027802 '+str(N)+' '+str(K+1)+' '+str(K+1)+'\n')


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

#########################################################################

# signal to noise ratio
SNR=0.5
do_images=False
# simulation
for ci in range(Nf):
  MS='L_SB'+str(ci)+'.MS'
  #os.system(sagecal+' -d '+MS+' -s sky0.txt -c cluster0.txt -t '+str(Tdelta)+' -O DATA -a 1 -B 2 -E 1 -p '+MS+'.S.solutions')
  os.system(sagecal+' -d '+MS+' -s sky0.txt -c cluster0.txt -t '+str(Tdelta)+' -O DATA -a 1 -B 2 -E 1')
  os.system('python addnoise.py '+MS+' '+str(SNR))
  if do_images:
    os.system(excon+' -m '+MS+' -p 8 -x 2 -c DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 12000 > /dev/null')

# calibration
os.system('mpirun -np 3 '+sagecal_mpi+' -f \'L_SB*.MS\'  -A 30 -P 2 -s sky.txt -c cluster.txt -I DATA -O MODEL_DATA -p zsol -G admm_rho.txt -n 4 -t '+str(Tdelta)+' -V')


for ci in range(Nf):
  MS='L_SB'+str(ci)+'.MS'
  if do_images:
    os.system(excon+' -m '+MS+' -p 8 -x 2 -c MODEL_DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 12000 -Q residual > /dev/null')

# create average images
if do_images:
  os.system('bash ./calmean.sh \'L_SB*.MS_I*fits\' 1 && python calmean_.py && mv bar.fits data.fits')
  os.system('bash ./calmean.sh \'L_SB*.MS_residual_I*fits\' 1 && python calmean_.py && mv bar.fits residual.fits')


#########################################################################
# Get the ra,dec coords of each cluster for imaging
cluster_ra,cluster_dec=get_cluster_centers(outskymodel1,outcluster1,ra0,dec0)
ignorelist='ignorelist.txt' # which clusters to ignore when simulating each cluster
rho=read_rho(initialrho,K)
for ci in range(Nf): #Nf
  MS='L_SB'+str(ci)+'.MS'
  freq=freqlist[ci]
  solutionfile=MS+'.solutions'
  tt=ctab.table(MS,readonly=True)  
  t1=tt.query(sortlist='TIME,ANTENNA1,ANTENNA2',columns='ANTENNA1,ANTENNA2,UVW,MODEL_DATA')
  vl=t1.getcol('MODEL_DATA')
  a1=t1.getcol('ANTENNA1')
  a2=t1.getcol('ANTENNA2')
  uvw=t1.getcol('UVW')
  nrtime=t1.nrows()
  assert(nrtime==B*Ts*Tdelta+N*Ts*Tdelta)
  XX=np.zeros((B*Ts*Tdelta),dtype=np.csingle)
  XY=np.zeros((B*Ts*Tdelta),dtype=np.csingle)
  YX=np.zeros((B*Ts*Tdelta),dtype=np.csingle)
  YY=np.zeros((B*Ts*Tdelta),dtype=np.csingle)
  uu=np.zeros((B*Ts*Tdelta),dtype=np.float32)
  vv=np.zeros((B*Ts*Tdelta),dtype=np.float32)
  ww=np.zeros((B*Ts*Tdelta),dtype=np.float32)
  ck=0
  for nr in range(0,nrtime):
      if (a1[nr]!=a2[nr]):
          XX[ck]=vl[nr,0,0]
          XY[ck]=vl[nr,0,1]
          YX[ck]=vl[nr,0,2]
          YY[ck]=vl[nr,0,3]
          uu[ck]=uvw[nr,0]
          vv[ck]=uvw[nr,1]
          ww[ck]=uvw[nr,2]
          ck+=1
  tt.close()
  
  # read solutions
  freqout,J=readsolutions(solutionfile)
  assert(J.shape[0]==K)
  assert(J.shape[1]==Ts*2*N)
  assert(freqout==freq)
  # predict sky model
  Ko,Ct=skytocoherencies_uvw(outskymodel1,outcluster1,uu,vv,ww,N,freqout,ra0,dec0)
  assert(Ko==K)
  assert(B*Ts*Tdelta==Ct.shape[1])

  J_norm,C_norm,Inf_mean=analysis_uvw_perdir(XX,XY,YX,YY,J,Ct,rho,freqlist,freqout,0.001,ra0,dec0,N,K,Ts,Tdelta,Nparallel=4)
  #for ck in range(K):
  #  os.system('python writecorr.py '+MS+' fff_'+str(ck))
  #  os.system(excon+' -x 0 -c CORRECTED_DATA -d 128 -p 20 -Q '+str(ck)+' -m '+MS+' > /dev/null')
  for ck in range(K):
      print('clus=%d sep=%f az=%f el=%f ||J||=%f ||C||=%f |Inf|=%f'%(ck,separation[ck],azimuth[ck],elevation[ck],J_norm[ck],C_norm[ck],Inf_mean[ck]))

  # make images while simulating each cluster (using the solutions)
  for ck in range(K):
      ff=open(ignorelist,'w+')
      for ck1 in range(K):
          if ck!=ck1:
              ff.write(str(ck1)+'\n')
      ff.close()
      hh,mm,ss=radToRA(cluster_ra[ck])
      dd,dmm,dss=radToDec(cluster_dec[ck])
      os.system(sagecal+' -d '+MS+' -s sky.txt -c cluster.txt -t '+str(Tdelta)+' -O DATA -a 1 -B 2 -E 1 -g '+ignorelist) # instead of using the solutions, use beam model
      os.system(excon+' -m '+MS+' -p 4 -x 2 -c DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 2400 -P '+str(hh)+','+str(mm)+','+str(ss)+','
        +str(dd)+','+str(dmm)+','+str(dss)+' -Q clus_'+str(ck)+' > /dev/null')

