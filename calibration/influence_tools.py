import math,sys,uuid
import numpy as np
import numpy.matlib
from multiprocessing import Pool
from multiprocessing import shared_memory
from casacore.measures import measures
from casacore.quanta import quantity
from calibration_tools import *

# return angle of separation of each clusrter in cluster file
# separation from ra0,dec0
# also check if source is below horizon, negative separation
# measure: casacore measure at core position at t0
# returns: separations,azimuths,elevations: Kx1 (degrees)
def calculate_separation(skymodel,clusterfile,ra0,dec0,measure):
  fh=open(skymodel,'r')
  fullset=fh.readlines()
  fh.close()
  S={}
  for cl in fullset:
   if (not cl.startswith('#')) and len(cl)>1:
     cl1=cl.split()
     S[cl1[0]]=cl1[1:]

  fh=open(clusterfile,'r')
  fullset=fh.readlines()
  fh.close()

  # determine number of clusters
  ci=0
  for cl in fullset:
   if (not cl.startswith('#')) and len(cl)>1:
     ci +=1
  K=ci

  ra0_q=quantity(ra0,'rad')
  dec0_q=quantity(dec0,'rad')
  target=measure.direction('j2000',ra0_q,dec0_q)

  separations=np.zeros(K,dtype=np.float32)
  azimuths=np.zeros(K,dtype=np.float32)
  elevations=np.zeros(K,dtype=np.float32)
  ck=0
  for cl in fullset:
   if (not cl.startswith('#')) and len(cl)>1:
     cl1=cl.split()
     for sname in cl1[2:3]: # only consider the first source of each cluster
       # 3:ra 3:dec sI 0 0 0 sP 0 0 0 0 0 0 freq0
       sinfo=S[sname]
       mra=(float(sinfo[0])+float(sinfo[1])/60.+float(sinfo[2])/3600.)*360./24.*math.pi/180.0
       mdec=(float(sinfo[3])+float(sinfo[4])/60.+float(sinfo[5])/3600.)*math.pi/180.0
       mra_q=quantity(mra,'rad')
       mdec_q=quantity(mdec,'rad')
       cluster_dir=measure.direction('j2000',mra_q,mdec_q)
       if ck<K-1:
         cluster_dir=measure.direction('j2000',mra_q,mdec_q)
       else: # last cluster is target
         cluster_dir=measure.direction('j2000',ra0_q,dec0_q)
       separation=measure.separation(target,cluster_dir)
       separations[ck]=separation.get_value()
       # get elevation of this dir
       azel=measure.measure(cluster_dir,'AZEL')
       azimuths[ck]=azel['m0']['value']/math.pi*180
       elevations[ck]=azel['m1']['value']/math.pi*180
     ck+=1

  return separations,azimuths,elevations


# return ra,dec of each cluster
# last cluster is set to ra0,dec0
# returns: ra,dec Kx1 (rad)
def get_cluster_centers(skymodel,clusterfile,ra0,dec0):
  fh=open(skymodel,'r')
  fullset=fh.readlines()
  fh.close()
  S={}
  for cl in fullset:
   if (not cl.startswith('#')) and len(cl)>1:
     cl1=cl.split()
     S[cl1[0]]=cl1[1:]

  fh=open(clusterfile,'r')
  fullset=fh.readlines()
  fh.close()

  # determine number of clusters
  ci=0
  for cl in fullset:
   if (not cl.startswith('#')) and len(cl)>1:
     ci +=1
  K=ci

  ra_q=np.zeros(K,dtype=np.float32)
  dec_q=np.zeros(K,dtype=np.float32)

  ck=0
  for cl in fullset:
   if (not cl.startswith('#')) and len(cl)>1:
     cl1=cl.split()
     for sname in cl1[2:3]: # only consider the first source of each cluster
       # 3:ra 3:dec sI 0 0 0 sP 0 0 0 0 0 0 freq0
       sinfo=S[sname]
       mra=(float(sinfo[0])+float(sinfo[1])/60.+float(sinfo[2])/3600.)*360./24.*math.pi/180.0
       mdec=(float(sinfo[3])+float(sinfo[4])/60.+float(sinfo[5])/3600.)*math.pi/180.0
       ra_q[ck]=mra
       dec_q[ck]=mdec
     ck+=1

  ra_q[K-1]=ra0
  dec_q[K-1]=dec0
  return ra_q,dec_q


def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result

# returns
# norm(J): Kx1
# norm(C): Kx1
# |mean(Influence)|: Kx1
def analysis_uvw_perdir(XX,XY,YX,YY,J,Ct,rho,freqs,freq,alpha,ra0,dec0,N,K,Ts,Tdelta,Nparallel=4):
    # XX,XY,YX,YY: data arrays : nrows x 1
    # J: solutions K x Nsol x 2, Nsol=2*N*Ts
    # Ct: coherencies K x nrows x 4
    # rho: rho values Kx1
    # freqs: frequencies Nfx1
    # freq: which frequency 
    # alpha: spatial constraint regularization parameter
    # N: stations
    # K: directions
    # Ts: how many calibrations
    # Tdelta: how many timeslots to use per calibration (-t option)
    # Nparallel=number of parallel jobs to use

    # baselines
    B=N*(N-1)//2

    if Ts<Nparallel:
        Nparallel=Ts
    
    # if 1, IQUV, else only I
    fullpol=0
    
    Ne=2 # consensus poly terms, same as -P parameter in sagecal
    polytype=1 # 0: ordinary, 1: Bernstein
    
    # create shared memory equal to XX,XY,YX,YY (times K )
    # buffers for parallel processing
    shmXX=shared_memory.SharedMemory(create=True,size=K*XX.nbytes)
    shmXY=shared_memory.SharedMemory(create=True,size=K*XY.nbytes)
    shmYX=shared_memory.SharedMemory(create=True,size=K*YX.nbytes)
    shmYY=shared_memory.SharedMemory(create=True,size=K*YY.nbytes)


    xx_shape=XX.shape
    xx0_shape=(K,xx_shape[0])

    shmLLR=shared_memory.SharedMemory(create=True,size=K*xx_shape[0]*np.dtype(np.float32).itemsize)
    # create arrays that can be used in multiprocessing
    XX0=np.ndarray(xx0_shape,dtype=XX.dtype,buffer=shmXX.buf)
    XY0=np.ndarray(xx0_shape,dtype=XY.dtype,buffer=shmXY.buf)
    YX0=np.ndarray(xx0_shape,dtype=YX.dtype,buffer=shmYX.buf)
    YY0=np.ndarray(xx0_shape,dtype=YY.dtype,buffer=shmYY.buf)

    LLR=np.ndarray((K,xx_shape[0]),dtype=np.float32,buffer=shmLLR.buf)

    # which frequency index to work with
    fidx=np.argmin(np.abs(freqs-freq))
    
    # addition to Hessian
    Hadd=np.zeros((K,4*N,4*N),dtype=np.float32)
    for ci in range(K):
     # note: F is dependent on rho when alpha!=0 
     # example: making F=rand(2N,2N) makes performance worse
     F,P=consensus_poly(Ne,N,freqs,freq,fidx,polytype=polytype,rho=rho[ci],alpha=alpha)
     FF=np.matmul(F.transpose(),F)
     if alpha>0.0:
       PP=np.matmul(P.transpose(),P)
       H11=0.5*rho[ci]*FF+0.5*alpha*rho[ci]*rho[ci]*PP
       H12=0.5*FF+0.5*alpha*rho[ci]*PP
       H21=H12
       H22=-0.5/rho[ci]*(np.eye(2*N)-FF)+0.5*alpha*PP
       Htilde=H11-np.matmul(H12,np.matmul(np.linalg.pinv(H22),H21))
       Hadd[ci]=np.kron(np.eye(2),Htilde)
     else:
       Hadd[ci]=0.5*rho[ci]*np.kron(np.eye(2),np.matmul(FF,np.eye(2*N)+np.matmul(np.linalg.pinv(np.eye(2*N)-FF),FF)))

############################# loop over timeslots
############################# local function
    @globalize
    def process_chunk(ncal):
        ts=ncal*Tdelta
        R=np.zeros((2*B*Tdelta,2),dtype=np.csingle)
        R[0:2*B*Tdelta:2,0]=XX[ts*B:ts*B+B*Tdelta]
        R[0:2*B*Tdelta:2,1]=XY[ts*B:ts*B+B*Tdelta]
        R[1:2*B*Tdelta:2,0]=YX[ts*B:ts*B+B*Tdelta]
        R[1:2*B*Tdelta:2,1]=YY[ts*B:ts*B+B*Tdelta]
       
        # D_Jgrad K x 4Nx4N tensor
        H=Hessianres(R,Ct[:,ts*B:ts*B+B*Tdelta],J[:,ncal*2*N:ncal*2*N+2*N],N)
        H+=Hadd
       
        # set to zero
        XX0[:,ts*B:ts*B+B*Tdelta]=0
        XY0[:,ts*B:ts*B+B*Tdelta]=0
        YX0[:,ts*B:ts*B+B*Tdelta]=0
        YY0[:,ts*B:ts*B+B*Tdelta]=0
       
        # dJ: 8 x K x 4NxB tensor
        dJ=Dsolutions_r(Ct[:,ts*B:ts*B+B*Tdelta],J[:,ncal*2*N:ncal*2*N+2*N],N,H)
        # dR: 8 x K x 4B x B (for all K)
        dR=Dresiduals_rk(Ct[:,ts*B:ts*B+B*Tdelta],J[:,ncal*2*N:ncal*2*N+2*N],N,dJ,0) # 0 for not adding I to dR
        # find mean value over columns
        for ck in range(K):
          for r in range(8):
            dR11=np.mean(dR[r,ck,0:4*B:4],axis=0)
            dR11=np.squeeze(np.matlib.repmat(dR11,1,Tdelta))
            XX0[ck,ts*B:ts*B+B*Tdelta] +=dR11
            dR11=np.mean(dR[r,ck,3:4*B:4],axis=0)
            dR11=np.squeeze(np.matlib.repmat(dR11,1,Tdelta))
            YY0[ck,ts*B:ts*B+B*Tdelta] +=dR11
            if fullpol:
              dR11=np.mean(dR[r,ck,1:4*B:4],axis=0)
              dR11=np.squeeze(np.matlib.repmat(dR11,1,Tdelta))
              XY0[ck,ts*B:ts*B+B*Tdelta] +=dR11
              dR11=np.mean(dR[r,ck,2:4*B:4],axis=0)
              dR11=np.squeeze(np.matlib.repmat(dR11,1,Tdelta))
              YX0[ck,ts*B:ts*B+B*Tdelta] +=dR11

        LLR[:,ts*B:ts*B+B*Tdelta]=log_likelihood_ratio(R,Ct[:,ts*B:ts*B+B*Tdelta],J[:,ncal*2*N:ncal*2*N+2*N],N)
############################# end local function
############################# loop over timeslots

    # create pool
    pool=Pool(Nparallel)
    pool.map(process_chunk,range(Ts))
    pool.close()
    pool.join()

    # scale by 8*(N*(N-1)/2)*T    
    scalefactor=8*(N*(N-1)/2)*Tdelta 
    XX0 *=scalefactor
    XY0 *=scalefactor
    YX0 *=scalefactor
    YY0 *=scalefactor

    J_norm=np.zeros(K,dtype=np.float32)
    C_norm=np.zeros(K,dtype=np.float32)
    Inf_mean=np.zeros(K,dtype=np.float32)
    llr_mean=np.zeros(K,dtype=np.float32)
    for ck in range(K):
        writeuvw('fff_'+str(ck),XX0[ck,:],XY0[ck,:],
                YX0[ck,:],YY0[ck,:])
        meaninfluence=np.abs(np.mean(XX0[ck,:])+np.mean(YY0[ck,:]))
        J_norm[ck]=np.linalg.norm(J[ck])
        C_norm[ck]=np.linalg.norm(Ct[ck])
        Inf_mean[ck]=meaninfluence
        llr_mean[ck]=np.mean(LLR[ck])


    # release shared memory
    shmXX.close()
    shmXX.unlink()
    shmXY.close()
    shmXY.unlink()
    shmYX.close()
    shmYX.unlink()
    shmYY.close()
    shmYY.unlink()
    shmLLR.close()
    shmLLR.unlink()

    return J_norm,C_norm,Inf_mean,llr_mean
