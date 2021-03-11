import math
import numpy as np
import numpy.matlib
from calibration_tools import *


def analysis_uvwdir_loop(skymodel,clusterfile,uvwfile,rhofile,solutionsfile):
    # stations
    N=62
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
    Ne=3 # poly order
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
    # how many timeslots to use per calibration (-t option)
    T=10
    Ts=int(XX.shape[0]//(B*T))

    # check this agrees with solutions
    nx,ny=J[0].shape
    if nx<2*N*Ts:
     print('Error: solutions size does not match with data size')
     exit

    
    # which frequency index to work with
    fidx=np.argmin(np.abs(f-freq))
    # note: F not dependent on rho, because it cancels out
    F=consensus_poly(Ne,N,f,f0,fidx,polytype=polytype)
    # example: making F=rand(2N,2N) makes performance worse
    
    # addition to Hessian
    Hadd=np.zeros((K,4*N,4*N),dtype=np.float32)
    FF=np.matmul(F.transpose(),F)
    for ci in range(K):
     Hadd[ci]=0.5*rho[ci]*np.kron(np.eye(2),np.matmul(FF,np.eye(2*N)+np.matmul(np.linalg.pinv(np.eye(2*N)-FF),FF)))

############################# loop over timeslots
    ts=0
    for ncal in range(Ts):
     R=np.zeros((2*B*T,2),dtype=np.csingle)
     R[0:2*B*T:2,0]=XX[ts*B:ts*B+B*T]
     R[0:2*B*T:2,1]=XY[ts*B:ts*B+B*T]
     R[1:2*B*T:2,0]=YX[ts*B:ts*B+B*T]
     R[1:2*B*T:2,1]=YY[ts*B:ts*B+B*T]
    
     # D_Jgrad K x 4Nx4N tensor
     H=Hessianres(R,Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N)
     H+=Hadd
    
     # set to zero
     XX[ts*B:ts*B+B*T]=0
     XY[ts*B:ts*B+B*T]=0
     YX[ts*B:ts*B+B*T]=0
     YY[ts*B:ts*B+B*T]=0
    
     if loop_in_r:
       for r in range(8):
         # dJ: K x 4NxB tensor
         dJ=Dsolutions(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,H,r)
         # dR: 4B x B (sum up all K)
         dR=Dresiduals(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,dJ,0,r) # 0 for not adding I to dR
         # find mean value over columns
         dR11=np.mean(dR[0:4*B:4],axis=0)
         dR11=np.squeeze(np.matlib.repmat(dR11,1,T))
         XX[ts*B:ts*B+B*T] +=dR11
         dR11=np.mean(dR[3:4*B:4],axis=0)
         dR11=np.squeeze(np.matlib.repmat(dR11,1,T))
         YY[ts*B:ts*B+B*T] +=dR11
     else:
       # dJ: 8 x K x 4NxB tensor
       dJ=Dsolutions_r(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,H)
       # dR: 8 x 4B x B (sum up all K)
       dR=Dresiduals_r(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,dJ,0) # 0 for not adding I to dR
       # find mean value over columns
       for r in range(8):
         dR11=np.mean(dR[r,0:4*B:4],axis=0)
         dR11=np.squeeze(np.matlib.repmat(dR11,1,T))
         XX[ts*B:ts*B+B*T] +=dR11
         dR11=np.mean(dR[r,3:4*B:4],axis=0)
         dR11=np.squeeze(np.matlib.repmat(dR11,1,T))
         YY[ts*B:ts*B+B*T] +=dR11
  
    
     print('%d %d %d'%(ts,Ts,ncal))
     ts +=T
############################# loop over timeslots

    scalefactor=8*(N*(N-1)/2)*T
    # scale by 8*(N*(N-1)/2)*T
    writeuvw('fff',scalefactor*XX,XY,YX,scalefactor*YY)




if __name__ == '__main__':
  # args skymodel clusterfile uvwfile rhofile solutionsfile
  import sys
  argc=len(sys.argv)
  if argc>5:
   analysis_uvwdir_loop(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
  else:
   print("Usage: python %s skymodel clusterfile uvwfile rhofile solutionsfile"%(sys.argv[0]))

  exit()

