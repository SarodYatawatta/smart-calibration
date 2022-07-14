import math
import subprocess as sb
import time
import numpy as np
import casacore.tables as ctab
from casacore.measures import measures
from casacore.quanta import quantity
from calibration_tools import *
from influence_tools import analysis_uvw_perdir,calculate_separation,calculate_separation_vec,get_cluster_centers
from astropy.io import fits
import astropy.time as atime

# executables
makems_binary='/home/sarod/scratch/software/bin/makems'
sagecal='/home/sarod/work/DIRAC/sagecal/build/dist/bin/sagecal_gpu'
sagecal_mpi='/home/sarod/work/DIRAC/sagecal/build/dist/bin/sagecal-mpi_gpu'
# imagers:
excon='/home/sarod/work/excon/src/MS/excon'
# or, set either of one to NULL to use the other
WSCLEAN='/usr/bin/wsclean'
# DP3 with necessary environment settings
DP3='export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/test/lib && export OPENBLAS_NUM_THREADS=1 && /home/sarod/scratch/software/bin/DP3'
# LINC script to download target sky
LINC_GET_TARGET='/home/sarod/scratch/LINC/scripts/download_skymodel_target.py --Radius 5'

# Spider settings
#makems_binary='/home/rapthor-syatawatta/bin/makems'
#sagecal='/home/rapthor-syatawatta/bin/sagecal'
#sagecal_mpi='/home/rapthor-syatawatta/bin/sagecal-mpi'
#excon='/home/rapthor-syatawatta/bin/excon'


# LOFAR core coords
X0='3826896.235129999928176m'
Y0='460979.4546659999759868m'
Z0='5064658.20299999974668m'

# Simulate a LOFAR observation, and generate training data
# K: directions for demixing + target
# input: K values of
#   influence map (normalized)
#   metadata: separation,az,el (degrees)
#   ||J||, ||C||, |Inf| (scalar, logarithm)
#   log likelihood ratio : scalar
#   frequency (lowest freq, logarithm)
# returns:
# x: input, shape: Kx(vector concatanation of the above), concatanated into a vector
# y: output, shape: K-1 vector of 1 or 0
def generate_training_data(Ninf=128):
    # Ninf: Influence map size (Ninf x Ninf)
    do_images=False
    do_solutions=False
    # number of frequencies
    Nf=3
    # HBA or LBA ?
    hba=(np.random.choice([0,1])==1)
    
    # epoch coordinate UTC 
    mydm=measures()
    x=X0
    y=Y0
    z=Z0
    mypos=mydm.position('ITRF',x,y,z)
    
    # Full time duration (slots), multiply with -t Tdelta option for full duration
    Ts=2
    Tdelta=10
    # integration time (s)
    Tint=1

    # approx A-Team coordinates, for generating targets close to one
    # CasA, CygA, HerA, TauA, VirA
    a_team_dirs=[(6.123273, 1.026748), (5.233838, 0.710912), (4.412048, 0.087195), (1.459697, 0.383912), (3.276019, 0.216299)]
    close_to_Ateam=-1 # 0,...4 will select one of the above
    distance_to_Ateam=1 # max distance, in degrees

    # strategy for sky model generation
    # 0: no special criteria (except target is above horizon)
    # 1: target has an outlier (== close_to_Ateam) at a distance (<= distance_to_Ateam)
    # 2: at least 2 outliers sources (except CasA/CygA) are close by
    sky_model_gen_strat=1

    valid_field=False
    # loop till we find a valid direction (above horizon) and epoch
    while not valid_field:
      # field coords (rad)
      if sky_model_gen_strat==0 or close_to_Ateam==-1:
        ra0=np.random.rand(1)*math.pi*2
        dec0=np.random.rand(1)*math.pi/2
        ra0=ra0[0]
        dec0=dec0[0]
      else: # generate direction close to given A-Team source
        # random distance in rad
        distance_from_here=np.random.rand(1)*distance_to_Ateam/180*math.pi
        ra0=a_team_dirs[close_to_Ateam][0]+distance_from_here[0]
        distance_from_here=np.random.rand(1)*distance_to_Ateam/180*math.pi
        dec0=a_team_dirs[close_to_Ateam][1]+distance_from_here[0]

      myra=quantity(str(ra0)+'rad')
      mydec=quantity(str(dec0)+'rad')
      mydir=mydm.direction('J2000',myra,mydec)
      t0=time.mktime(time.gmtime())+np.random.rand()*24*3600.0
      mytime=mydm.epoch('UTC',str(t0)+'s')
      mydm.doframe(mytime)
      mydm.doframe(mypos)
      # check elevation and field is above horizon, 5 deg above
      azel=mydm.measure(mydir,'AZEL')
      myel=azel['m1']['value']/math.pi*180

      # calculate separations
      separations=calculate_separation_vec(a_team_dirs,ra0,dec0,mydm)

      if sky_model_gen_strat==2:
        if ((separations[2]<=60 and separations[3]<=60) or
         (separations[3]<=60 and separations[4]<=60)) and myel>3.0:
          valid_field=True
      else:
        if myel>3.0:
          valid_field=True

    
    # now we have a valid ra0,dec0 and t0 tuple
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
    sb.run('cp '+makems_parset+' makems.cfg',shell=True)
    sb.run(makems_binary,shell=True)
    
    # output will be msout_p0
    msoutp0=msout+'_p0'
    
    sb.run('rsync -a ./LBA/FIELD '+msoutp0+'/',shell=True)
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
      sb.run('rsync -a ./HBA/LOFAR_ANTENNA_FIELD '+msoutp0+'/',shell=True)
    else:
      sb.run('rsync -a ./LBA/LOFAR_ANTENNA_FIELD '+msoutp0+'/',shell=True)
    
    # remove old files
    sb.run('rm -rf L_SB*.MS L_SB*fits',shell=True)
    # frequencies
    if hba:
        flow=110+np.random.rand()*(180-110)/2
        fhigh=110+(180-110)/2+np.random.rand()*(180-110)/2
    else:
        flow=30+np.random.rand()*(70-30)/2
        fhigh=30+(70-30)/2+np.random.rand()*(70-30)/2
    freqlist=np.linspace(flow,fhigh,num=Nf)*1e6
    
    f0=np.mean(freqlist)
    
    for ci in range(Nf):
        MS='L_SB'+str(ci)+'.MS'
        sb.run('rsync -a '+msoutp0+'/ '+MS,shell=True)
        sb.run('python changefreq.py '+MS+' '+str(freqlist[ci]),shell=True)
    
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
    
    # How to convert DP3 skymodel:
    # python ./convertmodel.py ../A-Team_lowres.skymodel base.sky base.cluster base.rho start_cluster_id
    # python ./convertmodel.py ../A-Team_lowres-update.skymodel base.sky base.cluster base.rho start_cluster_id
    sb.run('cp '+outskymodel+' tmp.sky',shell=True)
    sb.run('cat base.sky > '+outskymodel,shell=True)
    sb.run('cat tmp.sky >> '+outskymodel,shell=True)
    sb.run('cp '+outskymodel1+' tmp.sky',shell=True)
    sb.run('cat base.sky > '+outskymodel1,shell=True)
    sb.run('cat tmp.sky >> '+outskymodel1,shell=True)
    sb.run('cp '+outcluster+' tmp.cluster',shell=True)
    sb.run('cat base.cluster > '+outcluster,shell=True)
    sb.run('cat tmp.cluster >> '+outcluster,shell=True)
    sb.run('cp '+outcluster1+' tmp.cluster',shell=True)
    sb.run('cat base.cluster > '+outcluster1,shell=True)
    sb.run('cat tmp.cluster >> '+outcluster1,shell=True)
    sb.run('cp '+initialrho+' tmp.rho',shell=True)
    sb.run('cat base.rho > '+initialrho,shell=True)
    sb.run('cat tmp.rho >> '+initialrho,shell=True)
    
    # get separation of each cluster from target,
    # negative separation given if a cluster is below horizon
    separation,azimuth,elevation=calculate_separation(outskymodel1,outcluster1,ra0,dec0,mydm)
    
    #########################################################################
    # simulate errors for K directions, attenuate those errors
    # target = column K-1
    # outlier = columns 0..K-2
    if do_solutions:
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
    # signal to noise ratio: in the range 0.05 to 0.5
    SNR=np.random.rand()*(0.5-0.05)+0.05
    # simulation
    for ci in range(Nf):
      MS='L_SB'+str(ci)+'.MS'
      if do_solutions:
        sb.run(sagecal+' -d '+MS+' -s sky0.txt -c cluster0.txt -t '+str(Tdelta)+' -O DATA -a 1 -B 2 -E 1 -p '+MS+'.S.solutions',shell=True)
      else:
        sb.run(sagecal+' -d '+MS+' -s sky0.txt -c cluster0.txt -t '+str(Tdelta)+' -O DATA -a 1 -B 2 -E 1',shell=True)
      sb.run('python addnoise.py '+MS+' '+str(SNR),shell=True)
      if do_images:
        sb.run(excon+' -m '+MS+' -p 8 -x 2 -c DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 12000 > /dev/null',shell=True)
    
    # calibration, use --oversubscribe if not enough slots are available
    sb.run('mpirun -np 3 --oversubscribe '+sagecal_mpi+' -f \'L_SB*.MS\'  -A 30 -P 2 -s sky.txt -c cluster.txt -I DATA -O MODEL_DATA -p zsol -G admm_rho.txt -n 4 -t '+str(Tdelta)+' -V',shell=True)
    
    
    if do_images:
      for ci in range(Nf):
        MS='L_SB'+str(ci)+'.MS'
        sb.run(excon+' -m '+MS+' -p 8 -x 2 -c MODEL_DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 12000 -Q residual > /dev/null',shell=True)
    
    # create average images
    if do_images:
      sb.run('bash ./calmean.sh \'L_SB*.MS_I*fits\' 1 && python calmean_.py && mv bar.fits data.fits',shell=True)
      sb.run('bash ./calmean.sh \'L_SB*.MS_residual_I*fits\' 1 && python calmean_.py && mv bar.fits residual.fits',shell=True)
    
    
    #########################################################################
    # Get the ra,dec coords of each cluster for imaging
    cluster_ra,cluster_dec=get_cluster_centers(outskymodel1,outcluster1,ra0,dec0)
    ignorelist='ignorelist.txt' # which clusters to ignore when simulating each cluster
    rho=read_rho(initialrho,K)
    # sigma cutoff (this many std above) to detect sources, 
    # use a higher number than usual here
    sigma_threshold=7
    for ci in range(1): #Nf
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
      assert(abs(freqout-freq)<1e3)
      # predict sky model
      Ko,Ct=skytocoherencies_uvw(outskymodel1,outcluster1,uu,vv,ww,N,freqout,ra0,dec0)
      assert(Ko==K)
      assert(B*Ts*Tdelta==Ct.shape[1])
    
      J_norm,C_norm,Inf_mean,LLR_mean=analysis_uvw_perdir(XX,XY,YX,YY,J,Ct,rho,freqlist,freqout,0.001,ra0,dec0,N,K,Ts,Tdelta,Nparallel=4)
      for ck in range(K):
        proc1=sb.Popen('python writecorr.py '+MS+' fff_'+str(ck),shell=True)
        proc1.wait()
        proc1=sb.Popen(excon+' -x 0 -c CORRECTED_DATA -d '+str(Ninf)+' -p 20 -F 1e5,1e5,1e5,1e5 -Q inf_'+str(ck)+' -m '+MS+' -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C > /dev/null',shell=True)
        proc1.wait()
    
    sumpixels=np.zeros(K,dtype=np.float32)
    for ck in range(K-1): # only make images of outlier
      ff=open(ignorelist,'w+')
      for ck1 in range(K):
              if ck!=ck1:
                  ff.write(str(ck1)+'\n')
      ff.close()
      hh,mm,ss=radToRA(cluster_ra[ck])
      dd,dmm,dss=radToDec(cluster_dec[ck])
    
      # make images while simulating each cluster (using the solutions)
      fitsdata=np.zeros((1200,1200),dtype=np.float32)
      for ci in range(Nf):
          MS='L_SB'+str(ci)+'.MS'
          proc1=sb.Popen(sagecal+' -d '+MS+' -s sky.txt -c cluster.txt -t '+str(Tdelta)+' -O DATA -a 1 -B 2 -E 1 -g '+ignorelist,shell=True) # instead of using the solutions, use beam model
          proc1.wait()
          proc1=sb.Popen(excon+' -m '+MS+' -p 4 -x 0 -c DATA -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C -d 1200 -P '+str(hh)+','+str(mm)+','+str(ss)+','
            +str(dd)+','+str(dmm)+','+str(dss)+' -Q clus_'+str(ck)+' > /dev/null',shell=True)
          proc1.wait()
          hdu=fits.open(MS+'_clus_'+str(ck)+'_I.fits')
          fitsdata+=np.squeeze(hdu[0].data[0])
          hdu.close()
      fitsdata /= Nf
      # use four corners to find sigma
      fits_sigma=fitsdata[:200,:200].std()+fitsdata[-200:,:200].std()\
              +fitsdata[:200,-200:].std()+fitsdata[-200:,-200:].std()
      fits_sigma *=0.25
      fits_mask=fitsdata>sigma_threshold*fits_sigma
      masked_pix=fitsdata*fits_mask
      sumpixels[ck]+=masked_pix.sum()/(1+fits_mask.sum())
    
    # target flux, directly from the model
    sumpixels[K-1]=sum(sI)
    
    # Note: selection of sources based on flux may pickup false positives,
    # but better be safe than sorry
    print('cluster sep az el ||J|| ||C|| |Inf| LLR sI')
    for ck in range(K):
        print('%d %f %f %f %f %f %f %f %f'%(ck,separation[ck],azimuth[ck],elevation[ck],J_norm[ck],C_norm[ck],Inf_mean[ck],LLR_mean[ck],sumpixels[ck]))
    
    
    # set fluxes of sources below horizon (or below small degrees) to zero
    negel=elevation<3
    # this will set to 0 clusters below horizon
    sumpixels[negel]=0
    
    # If need to prioritize outlier sources to subtract:
    # see section 5 of
    # https://www.aanda.org/articles/aa/abs/2013/02/aa20874-12/aa20874-12.html,
    # it does not matter where the source is, its apparent intensity is the prime criterion.

    # create outputs, apply obvious constraints
    # label : 0,1 vector, exclude target
    y=sumpixels[:-1]
    y[sumpixels[:-1]>0]=1

    # input : NinfxNinf + (separation,azimuth,elevation) + log(||J||,||C||,|Inf|) + log_likelihood_ration+ lowest_frequency
    Nout=Ninf*Ninf+3+3+1+1
    x=np.zeros((K*Nout),dtype=np.float32)

    nfreq=0
    MS='L_SB'+str(nfreq)+'.MS'
    for ck in range(K):
       hdu=fits.open(MS+'_inf_'+str(ck)+'_I.fits')
       x[ck*Nout:(ck*Nout+Ninf*Ninf)]=np.reshape(np.squeeze(hdu[0].data[0]),(-1),order='F')
       hdu.close()
       imgnorm=np.linalg.norm(x[ck*Nout:(ck*Nout+Ninf*Ninf)])
       x[ck*Nout:(ck*Nout+Ninf*Ninf)] /= imgnorm
       # other data
       x[ck*Nout+Ninf*Ninf]=separation[ck]
       x[ck*Nout+Ninf*Ninf+1]=azimuth[ck]
       x[ck*Nout+Ninf*Ninf+2]=elevation[ck]
       x[ck*Nout+Ninf*Ninf+3]=np.log(J_norm[ck])
       x[ck*Nout+Ninf*Ninf+4]=np.log(C_norm[ck])
       x[ck*Nout+Ninf*Ninf+5]=np.log(Inf_mean[ck])
       #x[ck*Nout+Ninf*Ninf+5]=np.log(imgnorm)
       x[ck*Nout+Ninf*Ninf+6]=LLR_mean[ck]
       x[ck*Nout+Ninf*Ninf+7]=np.log(freqlist[nfreq])


    return x,y


# mslist: list of MS file names, out of which
# a subset will be chosen to sample a time duration of 'timesec' seconds
# channels of each MS will be averaged to one
# Nf=how many MS to exctract, equalt to the number of freqs used
# returns the list of extracted MS names
def extract_dataset(mslist,timesec,Nf=3):
   mslist.sort()
   msname=mslist[0]
   tt=ctab.table(msname,readonly=True)
   starttime= tt[0]['TIME']
   endtime=tt[tt.nrows()-1]['TIME']
   N=tt.nrows()
   tt.close()
   Nms=len(mslist)

   # need to have at least Nf MS
   assert(Nms>=Nf)

   # Parset for extracting and averaging
   parset_sample='extract_sample.parset'
   parset=open(parset_sample,'w+')

   # sample time interval
   t_start=np.random.rand()*(endtime-starttime)+starttime
   t_end=t_start+timesec
   t0=atime.Time(t_start/(24*60*60),format='mjd',scale='utc')
   dt=t0.to_datetime()
   str_tstart=str(dt.year)+'/'+str(dt.month)+'/'+str(dt.day)+'/'+str(dt.hour)+':'+str(dt.minute)+':'+str(dt.second)
   t0=atime.Time(t_end/(24*60*60),format='mjd',scale='utc')
   dt=t0.to_datetime()
   str_tend=str(dt.year)+'/'+str(dt.month)+'/'+str(dt.day)+'/'+str(dt.hour)+':'+str(dt.minute)+':'+str(dt.second)

   parset.write('steps=[avg]\n'
     +'avg.type=average\n'
     +'avg.timestep=1\n'
     +'avg.freqstep=64\n'
  #   +'msin.starttimeslot='+str(np.random.randint(N))+'\n'
  #   +'msin.ntimes='+str(60)+'\n'
     +'msin.datacolumn=DATA\n'
     +'msin.starttime='+str_tstart+'\n'
     +'msin.endtime='+str_tend+'\n')
   parset.close()

   # process subset of MS from mslist
   submslist=list()
   submslist.append(mslist[0])
   aa=list(np.random.choice(np.arange(1,Nms-1),Nf-2,replace=False))
   aa.sort()
   for ms_id in aa:
     submslist.append(mslist[ms_id])
   submslist.append(mslist[-1])

   # remove old files
   sb.run('rm -rf L_SB*.MS',shell=True)
   # now process each of selected MS
   extracted_ms=list()
   for ci in range(Nf):
      MS='L_SB'+str(ci)+'.MS'
      proc1=sb.Popen(DP3+' '+parset_sample+' msin='+submslist[ci]+' msout='+MS, shell=True)
      proc1.wait()
      extracted_ms.append(MS)

   # return extracted MS list
   return extracted_ms


# mslist: list of MS file names, out of which
# a subset will be chosen to sample a time duration of 'timesec' seconds
# channels of each MS will be averaged to one
# K: directions for demixing + target
# input: K values of
#   influence map (normalized)
#   metadata: separation,az,el (degrees)
#   ||J||, ||C||, |Inf| (scalar, logarithm)
#   log likelihood ratio : scalar
#   frequency (lowest freq, logarithm)
# returns:
# x: input, shape: Kx(vector concatanation of the above), concatanated into a vector
def get_info_from_dataset(mslist,timesec,Ninf=128):
    Nf=3
    K=6 # total must match = (A-team clusters + 1)
    submslist=extract_dataset(mslist,timesec,Nf)
    # get frequencies
    freqlist=np.zeros(Nf)
    for ci in range(len(submslist)):
        msname=submslist[ci]
        tf=ctab.table(msname+'/SPECTRAL_WINDOW',readonly=True)
        ch0=tf.getcol('CHAN_FREQ')
        reffreq=tf.getcol('REF_FREQUENCY')
        freqlist[ci]=ch0[0,0]
        tf.close()

    # update data with weights
    tt=ctab.table(submslist[0],readonly=True)
    data=tt.getcol('DATA')
    scalefac=np.sqrt(np.linalg.norm(data)/data.size)
    tt.close()
    for ci in range(len(submslist)):
      msname=submslist[ci]
      tt=ctab.table(msname,readonly=False)
      data=tt.getcol('DATA')
      data /=scalefac
      tt.putcol('DATA',data)
      tt.close()
      # add extra columns
      add_column(msname,'MODEL_DATA')
      add_column(msname,'CORRECTED_DATA')


    # Get target coords
    field=ctab.table(submslist[0]+'/FIELD',readonly=True)
    phase_dir=field.getcol('PHASE_DIR')
    ra0=phase_dir[0][0][0]
    dec0=phase_dir[0][0][1]
    field.close()

    # get antennas
    tt=ctab.table(submslist[0]+'/ANTENNA',readonly=True)
    N=tt.nrows()
    tt.close()
    # baselines
    B=N*(N-1)//2

    # get integration time
    tt=ctab.table(submslist[0],readonly=True)
    tt1=tt.getcol('INTERVAL')
    Tdelta=tt[0]['INTERVAL']
    t0=tt[0]['TIME']
    tt.close()
    Tslots=math.ceil(timesec/Tdelta)

    # epoch coordinate UTC 
    mydm=measures()
    x=X0
    y=Y0
    z=Z0
    mypos=mydm.position('ITRF',x,y,z)
    mytime=mydm.epoch('UTC',str(t0)+'s')
    mydm.doframe(mytime)
    mydm.doframe(mypos)

    # download target sky model (using LINC script) (including path)
    target_skymodel='./target.sky.txt'
    sb.run('rm -rf '+target_skymodel,shell=True)
    sb.run('python '+LINC_GET_TARGET+' '+submslist[0]+' '+target_skymodel,shell=True)

    outskymodel='sky.txt' # for calibration
    outcluster='cluster.txt' # for calibration
    initialrho='admm_rho.txt' # values for rho, determined analytically

    # Convert DP3 skymodel of target
    sb.run('python ./convertmodel.py '+target_skymodel+' tmp.sky tmp.cluster tmp.rho 1',shell=True)
    sb.run('cat base.sky > '+outskymodel,shell=True)
    sb.run('cat tmp.sky >> '+outskymodel,shell=True)
    sb.run('cat base.cluster > '+outcluster,shell=True)
    sb.run('cat tmp.cluster >> '+outcluster,shell=True)
    sb.run('cat base.rho > '+initialrho,shell=True)
    sb.run('cat tmp.rho >> '+initialrho,shell=True)

    separation,azimuth,elevation=calculate_separation(outskymodel,outcluster,ra0,dec0,mydm)
    # Full time duration =timesec
    # calibration duration (slots)
    Tdelta=10
    Ts=(Tslots+Tdelta-1)//Tdelta

    # calibration, use --oversubscribe if not enough slots are available
    sb.run('mpirun -np 3 --oversubscribe '+sagecal_mpi+' -f \'L_SB*.MS\'  -A 30 -P 2 -s sky.txt -c cluster.txt -I DATA -O MODEL_DATA -p zsol -G admm_rho.txt -n 4 -t '+str(Tdelta)+' -V',shell=True)
 
    #########################################################################
    # Get the ra,dec coords of each cluster for imaging
    cluster_ra,cluster_dec=get_cluster_centers(outskymodel,outcluster,ra0,dec0)
    rho=read_rho(initialrho,K)
    for ci in range(1): #Nf
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
      assert(nrtime==(B+N)*Tslots)
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
      assert(abs(freqout-freq)<1e3)
      # predict sky model
      Ko,Ct=skytocoherencies_uvw(outskymodel,outcluster,uu,vv,ww,N,freqout,ra0,dec0)
      assert(Ko==K)
      assert(B*Ts*Tdelta==Ct.shape[1])
    
      J_norm,C_norm,Inf_mean,LLR_mean=analysis_uvw_perdir(XX,XY,YX,YY,J,Ct,rho,freqlist,freqout,0.001,ra0,dec0,N,K,Ts,Tdelta,Nparallel=4)
      for ck in range(K):
        proc1=sb.Popen('python writecorr.py '+MS+' fff_'+str(ck),shell=True)
        proc1.wait()
        if WSCLEAN is not None:
          proc1=sb.Popen(WSCLEAN+' -data-column CORRECTED_DATA -size '+str(Ninf)+' '+str(Ninf)+' -scale 20asec -niter 0 -name '+MS+'_'+str(ck)+' '+MS,shell=True)
        else:
          proc1=sb.Popen(excon+' -x 0 -c CORRECTED_DATA -d '+str(Ninf)+' -p 20 -F 1e5,1e5,1e5,1e5 -Q inf_'+str(ck)+' -m '+MS+' -A /dev/shm/A -B /dev/shm/B -C /dev/shm/C > /dev/null',shell=True)
        proc1.wait()
    
    print('cluster sep az el ||J|| ||C|| |Inf| LLR')
    for ck in range(K):
        print('%d %f %f %f %f %f %f %f'%(ck,separation[ck],azimuth[ck],elevation[ck],J_norm[ck],C_norm[ck],Inf_mean[ck],LLR_mean[ck]))
    
    
    # input : NinfxNinf + (separation,azimuth,elevation) + log(||J||,||C||,|Inf|) + log_likelihood_ration+ lowest_frequency
    Nout=Ninf*Ninf+3+3+1+1
    x=np.zeros((K*Nout),dtype=np.float32)

    nfreq=0
    MS='L_SB'+str(nfreq)+'.MS'
    for ck in range(K):
       if WSCLEAN is not None:
         hdu=fits.open(MS+'_'+str(ck)+'-image.fits')
       else:
         hdu=fits.open(MS+'_inf_'+str(ck)+'_I.fits')
       x[ck*Nout:(ck*Nout+Ninf*Ninf)]=np.reshape(np.squeeze(hdu[0].data[0]),(-1),order='F')
       hdu.close()
       imgnorm=np.linalg.norm(x[ck*Nout:(ck*Nout+Ninf*Ninf)])
       x[ck*Nout:(ck*Nout+Ninf*Ninf)] /= (imgnorm + 1e-9)
       # other data
       x[ck*Nout+Ninf*Ninf]=separation[ck]
       x[ck*Nout+Ninf*Ninf+1]=azimuth[ck]
       x[ck*Nout+Ninf*Ninf+2]=elevation[ck]
       x[ck*Nout+Ninf*Ninf+3]=np.log(J_norm[ck])
       x[ck*Nout+Ninf*Ninf+4]=np.log(C_norm[ck])
       x[ck*Nout+Ninf*Ninf+5]=np.log(Inf_mean[ck])
       #x[ck*Nout+Ninf*Ninf+5]=np.log(imgnorm)
       x[ck*Nout+Ninf*Ninf+6]=LLR_mean[ck]
       x[ck*Nout+Ninf*Ninf+7]=np.log(freqlist[nfreq])

    return x


## adds an extra column to an MS
def add_column(msname,colname):
  tt=ctab.table(msname,readonly=False)
  cl=tt.getcol('DATA')
  (nrows,nchans,npols)=cl.shape
  vl=np.zeros(shape=cl.shape,dtype='complex64')
  dmi=tt.getdminfo('DATA')
  dmi['NAME']=colname
  mkd=ctab.maketabdesc(ctab.makearrcoldesc(colname,shape=np.array(np.zeros([nchans,npols])).shape,valuetype='complex',value=0.))
  tt.addcols(mkd,dmi)
  tt.putcol(colname,vl)
  tt.close()

# Simulate a LOFAR observation, and sky models
# Nf: number of frequencies
# returns: separation,azimuth,elevation (rad): Kx1 arrays
#  freq (Hz): lowest freq
def simulate_data(Nf=3,Tdelta=10):
    # K: directions for demixing + target
    K=6 # total must match = (A-team clusters + 1)
    do_images=False
    do_solutions=True
    # HBA or LBA ?
    hba=(np.random.choice([0,1])==1)

    # epoch coordinate UTC
    mydm=measures()
    x=X0
    y=Y0
    z=Z0
    mypos=mydm.position('ITRF',x,y,z)

    # Full time duration (slots), multiply with -t Tdelta option for full duration
    Ts=2
    # integration time (s)
    Tint=1

    # approx A-Team coordinates, for generating targets close to one
    # CasA, CygA, HerA, TauA, VirA
    a_team_dirs=[(6.123273, 1.026748), (5.233838, 0.710912), (4.412048, 0.087195), (1.459697, 0.383912), (3.276019, 0.216299)]
    close_to_Ateam=-1 # 0,...4 will select one of the above
    distance_to_Ateam=1 # max distance, in degrees

    # strategy for sky model generation
    # 0: no special criteria (except target is above horizon)
    # 1: target has an outlier (== close_to_Ateam) at a distance (<= distance_to_Ateam)
    # 2: at least 2 outliers sources (except CasA/CygA) are close by
    sky_model_gen_strat=2

    valid_field=False
    # loop till we find a valid direction (above horizon) and epoch
    while not valid_field:
      # field coords (rad)
      if sky_model_gen_strat==0 or close_to_Ateam==-1:
        ra0=np.random.rand(1)*math.pi*2
        dec0=np.random.rand(1)*math.pi/2
        ra0=ra0[0]
        dec0=dec0[0]
      else: # generate direction close to given A-Team source
        # random distance in rad
        distance_from_here=np.random.rand(1)*distance_to_Ateam/180*math.pi
        ra0=a_team_dirs[close_to_Ateam][0]+distance_from_here[0]
        distance_from_here=np.random.rand(1)*distance_to_Ateam/180*math.pi
        dec0=a_team_dirs[close_to_Ateam][1]+distance_from_here[0]

      myra=quantity(str(ra0)+'rad')
      mydec=quantity(str(dec0)+'rad')
      mydir=mydm.direction('J2000',myra,mydec)
      t0=time.mktime(time.gmtime())+np.random.rand()*24*3600.0
      mytime=mydm.epoch('UTC',str(t0)+'s')
      mydm.doframe(mytime)
      mydm.doframe(mypos)
      # check elevation and field is above horizon, 5 deg above
      azel=mydm.measure(mydir,'AZEL')
      myel=azel['m1']['value']/math.pi*180

      # calculate separations
      separations=calculate_separation_vec(a_team_dirs,ra0,dec0,mydm)

      if sky_model_gen_strat==2:
        if ((separations[2]<=60 and separations[3]<=60) or
         (separations[3]<=60 and separations[4]<=60)) and myel>3.0:
          valid_field=True
      else:
        if myel>3.0:
          valid_field=True


    # now we have a valid ra0,dec0 and t0 tuple
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
    sb.run('cp '+makems_parset+' makems.cfg',shell=True)
    sb.run(makems_binary,shell=True)

    # output will be msout_p0
    msoutp0=msout+'_p0'

    sb.run('rsync -a ./LBA/FIELD '+msoutp0+'/',shell=True)
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
      sb.run('rsync -a ./HBA/LOFAR_ANTENNA_FIELD '+msoutp0+'/',shell=True)
    else:
      sb.run('rsync -a ./LBA/LOFAR_ANTENNA_FIELD '+msoutp0+'/',shell=True)

    # remove old files
    sb.run('rm -rf L_SB*.MS L_SB*fits',shell=True)
    # frequencies
    if hba:
        flow=110+np.random.rand()*(180-110)/2
        fhigh=110+(180-110)/2+np.random.rand()*(180-110)/2
    else:
        flow=30+np.random.rand()*(70-30)/2
        fhigh=30+(70-30)/2+np.random.rand()*(70-30)/2
    freqlist=np.linspace(flow,fhigh,num=Nf)*1e6

    f0=np.mean(freqlist)

    for ci in range(Nf):
        MS='L_SB'+str(ci)+'.MS'
        sb.run('rsync -a '+msoutp0+'/ '+MS,shell=True)
        sb.run('python changefreq.py '+MS+' '+str(freqlist[ci]),shell=True)

    #########################################################################
    # sky model/error simulation

    # simulate target field and outlier, the remaining clusters are part of A-team
    # Sources (directions) used in calibration:
    # first one for center, 1,2,3,.. for outlier sources
    # and last one for weak sources (so minimum 2), 3 will be the weak sources

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

    # How to convert DP3 skymodel:
    # python ./convertmodel.py ../A-Team_lowres.skymodel base.sky base.cluster base.rho start_cluster_id
    # python ./convertmodel.py ../A-Team_lowres-update.skymodel base.sky base.cluster base.rho start_cluster_id
    sb.run('cp '+outskymodel+' tmp.sky',shell=True)
    sb.run('cat base.sky > '+outskymodel,shell=True)
    sb.run('cat tmp.sky >> '+outskymodel,shell=True)
    sb.run('cp '+outskymodel1+' tmp.sky',shell=True)
    sb.run('cat base.sky > '+outskymodel1,shell=True)
    sb.run('cat tmp.sky >> '+outskymodel1,shell=True)
    sb.run('cp '+outcluster+' tmp.cluster',shell=True)
    sb.run('cat base.cluster > '+outcluster,shell=True)
    sb.run('cat tmp.cluster >> '+outcluster,shell=True)
    sb.run('cp '+outcluster1+' tmp.cluster',shell=True)
    sb.run('cat base.cluster > '+outcluster1,shell=True)
    sb.run('cat tmp.cluster >> '+outcluster1,shell=True)
    sb.run('cp '+initialrho+' tmp.rho',shell=True)
    sb.run('cat base.rho > '+initialrho,shell=True)
    sb.run('cat tmp.rho >> '+initialrho,shell=True)

    separation,azimuth,elevation=calculate_separation(outskymodel1,outcluster1,ra0,dec0,mydm)
    #########################################################################
    # simulate errors for K directions, attenuate those errors
    # target = column K-1
    # outlier = columns 0..K-2
    if do_solutions:
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
    # signal to noise ratio: in the range 0.05 to 0.5
    SNR=np.random.rand()*(0.5-0.05)+0.05
    # simulation
    for ci in range(Nf):
      MS='L_SB'+str(ci)+'.MS'
      if do_solutions:
        sb.run(sagecal+' -d '+MS+' -s sky0.txt -c cluster0.txt -t '+str(Tdelta)+' -O DATA -a 1 -B 2 -E 1 -p '+MS+'.S.solutions',shell=True)
      else:
        sb.run(sagecal+' -d '+MS+' -s sky0.txt -c cluster0.txt -t '+str(Tdelta)+' -O DATA -a 1 -B 2 -E 1',shell=True)
      sb.run('python addnoise.py '+MS+' '+str(SNR),shell=True)
      if do_images:
        sb.run(excon+' -m '+MS+' -p 8 -x 2 -c DATA -d 12000 > /dev/null',shell=True)

    # create average images
    if do_images:
      sb.run('bash ./calmean.sh \'L_SB*.MS_I*fits\' 1 && python calmean_.py && mv bar.fits data.fits',shell=True)

    return separation,azimuth,elevation,freqlist[0],N
