import math,sys,uuid
import numpy as np
import numpy.matlib
from multiprocessing import Pool
from multiprocessing import shared_memory
from calibration_tools import *


def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result

def analysis_uvwdir_loop(skymodel,clusterfile,uvwfile,rhofile,solutionsfile,alpha,Nparallel=4):
    # alpha: spatial constraint regularization parameter
    # Nparallel=number of parallel jobs to use
    # stations
    N=62
    #GG N=61
    # baselines
    B=int(N*(N-1)/2)
    
#   skymodel='/tmp/sky.txt'
#    clusterfile='/tmp/cluster.txt'
#    uvwfile='./smalluvw.txt'
#    solutionsfile='/tmp/L_SB4.MS.solutions'
#    rhofile='/tmp/admm_rho.txt'
    
    # if 1, IQUV, else only I
    fullpol=0
    loop_in_r=False # use 8 blocks instead of looping
    
    # reference freq
    f0=150.0e6 # mean of all freqs
    #%%%%%%%%%%%%%%%%%% consensus polynomial info
    Nf=8 # no. of freqs: make sure to match all data
    f=np.linspace(115,185,Nf)*1e6
    Ne=3 # consensus poly terms, same as -P parameter in sagecal
    polytype=1 # 0: ordinary, 1: Bernstein
    ra0=0
    dec0=math.pi/2
    
    # read solutions file (also get the frequency(MHz)) J: Kx2N Nt x 2 (2Nx2 blocks Nt times)
    freq,J=readsolutions(solutionsfile)
    # read sky model Ct: Kx T x 4 (each row XX,XY,YX,YY)
    K,Ct=skytocoherencies(skymodel,clusterfile,uvwfile,N,freq,ra0,dec0)
    
    # ADMM rho, per each direction, scale later
    # scale rho linearly with sI
    rho=read_rho(rhofile,K)

    # read u,v,w,xx(re,im), xy(re,im) yx(re,im) yy(re,im)
    XX,XY,YX,YY=readuvw(uvwfile)
    # create shared memory equal to XX,XY,YX,YY buffers for parallel processing
    shmXX=shared_memory.SharedMemory(create=True,size=XX.nbytes)
    shmXY=shared_memory.SharedMemory(create=True,size=XY.nbytes)
    shmYX=shared_memory.SharedMemory(create=True,size=YX.nbytes)
    shmYY=shared_memory.SharedMemory(create=True,size=YY.nbytes)
    # create arrays that can be used in multiprocessing
    XX0=np.ndarray(XX.shape,dtype=XX.dtype,buffer=shmXX.buf)
    XY0=np.ndarray(XY.shape,dtype=XY.dtype,buffer=shmXY.buf)
    YX0=np.ndarray(YX.shape,dtype=YX.dtype,buffer=shmYX.buf)
    YY0=np.ndarray(YY.shape,dtype=YY.dtype,buffer=shmYY.buf)
    # how many timeslots to use per calibration (-t option)
    T=10
    #GG T=5
    Ts=int(XX.shape[0]//(B*T))

    # check this agrees with solutions
    nx,ny=J[0].shape
    if nx<2*N*Ts:
     print('Error: solutions size does not match with data size')
     exit

    
    # which frequency index to work with
    fidx=np.argmin(np.abs(f-freq))
    
    # addition to Hessian
    Hadd=np.zeros((K,4*N,4*N),dtype=np.float32)
    for ci in range(K):
     # note: F is dependent on rho when alpha!=0 
     # example: making F=rand(2N,2N) makes performance worse
     F,P=consensus_poly(Ne,N,f,f0,fidx,polytype=polytype,rho=rho[ci],alpha=alpha)
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
        ts=ncal*T
        print('%d %d %d'%(ts,Ts,ncal))
        R=np.zeros((2*B*T,2),dtype=np.csingle)
        R[0:2*B*T:2,0]=XX[ts*B:ts*B+B*T]
        R[0:2*B*T:2,1]=XY[ts*B:ts*B+B*T]
        R[1:2*B*T:2,0]=YX[ts*B:ts*B+B*T]
        R[1:2*B*T:2,1]=YY[ts*B:ts*B+B*T]
       
        # D_Jgrad K x 4Nx4N tensor
        H=Hessianres(R,Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N)
        H+=Hadd
       
        # set to zero
        XX0[ts*B:ts*B+B*T]=0
        XY0[ts*B:ts*B+B*T]=0
        YX0[ts*B:ts*B+B*T]=0
        YY0[ts*B:ts*B+B*T]=0
       
        if loop_in_r:
          for r in range(8):
            # dJ: K x 4NxB tensor
            dJ=Dsolutions(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,H,r)
            # dR: 4B x B (sum up all K)
            dR=Dresiduals(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,dJ,0,r) # 0 for not adding I to dR
            # find mean value over columns
            dR11=np.mean(dR[0:4*B:4],axis=0)
            dR11=np.squeeze(np.matlib.repmat(dR11,1,T))
            XX0[ts*B:ts*B+B*T] +=dR11
            dR11=np.mean(dR[3:4*B:4],axis=0)
            dR11=np.squeeze(np.matlib.repmat(dR11,1,T))
            YY0[ts*B:ts*B+B*T] +=dR11
        else:
          # dJ: 8 x K x 4NxB tensor
          dJ=Dsolutions_r(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,H)
          # dR: 8 x 4B x B (sum up all K)
          dR=Dresiduals_r(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,dJ,0) # 0 for not adding I to dR
          # find mean value over columns
          for r in range(8):
            dR11=np.mean(dR[r,0:4*B:4],axis=0)
            dR11=np.squeeze(np.matlib.repmat(dR11,1,T))
            XX0[ts*B:ts*B+B*T] +=dR11
            dR11=np.mean(dR[r,3:4*B:4],axis=0)
            dR11=np.squeeze(np.matlib.repmat(dR11,1,T))
            YY0[ts*B:ts*B+B*T] +=dR11
############################# end local function
############################# loop over timeslots

    # create pool
    pool=Pool(Nparallel)
    pool.map(process_chunk,range(Ts))
    pool.close()
    pool.join()

    # copy back from shared memory
    XX[:]=XX0[:]
    XY[:]=XY0[:]
    YX[:]=YX0[:]
    YY[:]=YY0[:]
    # release shared memory
    shmXX.close()
    shmXX.unlink()
    shmXY.close()
    shmXY.unlink()
    shmYX.close()
    shmYX.unlink()
    shmYY.close()
    shmYY.unlink()

    scalefactor=8*(N*(N-1)/2)*T 
    # scale by 8*(N*(N-1)/2)*T    
    writeuvw('fff',scalefactor*XX,XY,YX,scalefactor*YY)




if __name__ == '__main__':
  # args skymodel clusterfile uvwfile rhofile solutionsfile alpha parallel_jobs
  import sys
  argc=len(sys.argv)
  if argc==7:
   analysis_uvwdir_loop(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],float(sys.argv[6]))
  elif argc==8:
   analysis_uvwdir_loop(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],float(sys.argv[6]),int(sys.argv[7]))
  else:
   print("Usage: python %s skymodel clusterfile uvwfile rhofile solutionsfile alpha parallel_jobs"%(sys.argv[0]))

  exit()

