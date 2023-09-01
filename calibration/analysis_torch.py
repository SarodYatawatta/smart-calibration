import math,sys,uuid
import numpy as np
import numpy.matlib
import torch
from torch.multiprocessing import Pool,Process,set_start_method
from calibration_tools import *

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

def process_chunk(ncal,XX,XY,YX,YY,Ct,J,Hadd,T,Ts,B,N,loop_in_r,fullpol):
        ts=ncal*T
        print('%d %d %d'%(ts,Ts,ncal))
        R=torch.zeros((2*B*T,2),dtype=torch.cfloat).to(mydevice)
        R[0:2*B*T:2,0]=XX[ts*B:ts*B+B*T]
        R[0:2*B*T:2,1]=XY[ts*B:ts*B+B*T]
        R[1:2*B*T:2,0]=YX[ts*B:ts*B+B*T]
        R[1:2*B*T:2,1]=YY[ts*B:ts*B+B*T]
       
        # D_Jgrad K x 4Nx4N tensor
        H=Hessianres_torch(R,Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,mydevice)
        H+=Hadd
       
        # set to zero
        XX[ts*B:ts*B+B*T]=0
        XY[ts*B:ts*B+B*T]=0
        YX[ts*B:ts*B+B*T]=0
        YY[ts*B:ts*B+B*T]=0
       
        if loop_in_r:
          for r in range(8):
            # dJ: K x 4NxB tensor
            dJ=Dsolutions_torch(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,H,r,mydevice)
            # dR: 4B x B (sum up all K)
            dR=Dresiduals_torch(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,dJ,0,r,mydevice) # 0 for not adding I to dR
            # find mean value over columns
            dR11=torch.mean(dR[0:4*B:4],dim=0)
            dR11=torch.squeeze(dR11.repeat(1,T))
            XX[ts*B:ts*B+B*T] +=dR11
            dR11=torch.mean(dR[3:4*B:4],dim=0)
            dR11=torch.squeeze(dR11.repeat(1,T))
            YY[ts*B:ts*B+B*T] +=dR11
            if fullpol:
              dR11=torch.mean(dR[1:4*B:4],dim=0)
              dR11=torch.squeeze(dR11.repeat(1,T))
              XY[ts*B:ts*B+B*T] +=dR11
              dR11=torch.mean(dR[2:4*B:4],dim=0)
              dR11=torch.squeeze(dR11.repeat(1,T))
              YX[ts*B:ts*B+B*T] +=dR11
        else:
          # dJ: 8 x K x 4NxB tensor
          dJ=Dsolutions_r_torch(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,H,mydevice)
          # dR: 8 x 4B x B (sum up all K)
          dR=Dresiduals_r_torch(Ct[:,ts*B:ts*B+B*T],J[:,ncal*2*N:ncal*2*N+2*N],N,dJ,0,mydevice) # 0 for not adding I to dR
          # find mean value over columns
          for r in range(8):
            dR11=torch.mean(dR[r,0:4*B:4],dim=0)
            dR11=torch.squeeze(dR11.repeat(1,T))
            XX[ts*B:ts*B+B*T] +=dR11
            dR11=torch.mean(dR[r,3:4*B:4],dim=0)
            dR11=torch.squeeze(dR11.repeat(1,T))
            YY[ts*B:ts*B+B*T] +=dR11
            if fullpol:
              dR11=torch.mean(dR[r,1:4*B:4],dim=0)
              dR11=torch.squeeze(dR11.repeat(1,T))
              XY[ts*B:ts*B+B*T] +=dR11
              dR11=torch.mean(dR[r,2:4*B:4],dim=0)
              dR11=torch.squeeze(dR11.repeat(1,T))
              YX[ts*B:ts*B+B*T] +=dR11

        del R,H,dJ,dR,dR11

 
def analysis_uvwdir_loop(skymodel,clusterfile,uvwfile,rhofile,solutionsfile,z_solfile,flow=110,fhigh=170,ra0=0,dec0=math.pi/2,tslots=10,Nparallel=4):
    # ra0,dec0: phase center (rad)
    # tslots: -t option
    # Nparallel=number of parallel jobs to use

    # alpha: spatial constraint regularization parameter (also read from rhofile)
    flow=flow*1e6
    fhigh=fhigh*1e6
   
    # if 1, IQUV, else only I
    fullpol=0
    loop_in_r=False# use 8 blocks instead of looping
    
    # read Z (global) solutions to get metadata for consensus polynomial
    N,f0,Ne,K1,Z1=read_global_solutions(z_solfile)
    # N: stations
    # Ne: consensus poly terms, same as -P parameter in sagecal
    # baselines
    B=int(N*(N-1)/2)
    # reference freq (for consensus polynomial)
    assert(f0>=flow and f0<=fhigh) # mean of all freqs
    #%%%%%%%%%%%%%%%%%% consensus polynomial info
    Nf=8 # no. of freqs: make sure to match all data
    f=np.linspace(flow,fhigh,Nf)
    polytype=1 # 0: ordinary, 1: Bernstein
    
    # read solutions file (also get the frequency(MHz)) J: Kx2N Nt x 2 (2Nx2 blocks Nt times)
    freq,J=readsolutions(solutionsfile)
    # read sky model Ct: Kx T x 4 (each row XX,XY,YX,YY)
    K,Ct=skytocoherencies_torch(skymodel,clusterfile,uvwfile,N,freq,ra0,dec0,mydevice)
    assert(K==K1)
    
    # ADMM rho, per each direction, scale later
    # scale rho linearly with sI
    rho_spectral,rho_spatial=read_rho(rhofile,K)

    # read u,v,w,xx(re,im), xy(re,im) yx(re,im) yy(re,im)
    XX,XY,YX,YY=readuvw(uvwfile)
    # how many timeslots to use per calibration (-t option)
    T=tslots
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

    # map to torch
    XX=torch.from_numpy(XX).to(mydevice)
    XY=torch.from_numpy(XY).to(mydevice)
    YX=torch.from_numpy(YX).to(mydevice)
    YY=torch.from_numpy(YY).to(mydevice)
    Hadd=torch.from_numpy(Hadd).to(mydevice)
    J=torch.from_numpy(J).to(mydevice)
    for ci in range(K):
     # note: F is dependent on rho when alpha!=0 
     # example: making F=rand(2N,2N) makes performance worse
     alpha=rho_spatial[ci]
     F,P=consensus_poly(Ne,N,f,f0,fidx,polytype=polytype,rho=rho_spectral[ci],alpha=alpha)
     FF=np.matmul(F.transpose(),F)
     if alpha>0.0:
       PP=np.matmul(P.transpose(),P)
       H11=0.5*rho_spectral[ci]*FF+0.5*alpha*rho_spectral[ci]*rho_spectral[ci]*PP
       H12=0.5*FF+0.5*alpha*rho_spectral[ci]*PP
       H21=H12
       H22=-0.5/rho_spectral[ci]*(np.eye(2*N)-FF)+0.5*alpha*PP
       Htilde=H11-np.matmul(H12,np.matmul(np.linalg.pinv(H22),H21))
       Hadd[ci]=torch.from_numpy(np.kron(np.eye(2),Htilde)).to(mydevice)
     else:
       Hadd[ci]=torch.from_numpy(0.5*rho_spectral[ci]*np.kron(np.eye(2),np.matmul(FF,np.eye(2*N)+np.matmul(np.linalg.pinv(np.eye(2*N)-FF),FF)))).to(mydevice)

    XX.share_memory_()
    XY.share_memory_()
    YX.share_memory_()
    YY.share_memory_()

    # create pool
    pool=Pool(processes=Nparallel)
    argin=[(ci,XX,XY,YX,YY,Ct,J,Hadd,T,Ts,B,N,loop_in_r,fullpol) for ci in range(Ts)]
    pool.starmap(process_chunk,argin)
    pool.close()
    pool.join()

    # scale by 8*(N*(N-1)/2)*T    
    scalefactor=8*(N*(N-1)/2)*T 

    XX=XX*scalefactor
    XY=XY*scalefactor
    YX=YX*scalefactor
    YY=YY*scalefactor
    XX=XX.cpu().numpy()
    XY=XY.cpu().numpy()
    YX=YX.cpu().numpy()
    YY=YY.cpu().numpy()
    writeuvw('fff',XX,XY,YX,YY)


if __name__ == '__main__':
  # setup multiprocessing
  try:
    set_start_method('spawn')
  except RuntimeError:
    pass

  # args skymodel clusterfile uvwfile rhofile solutionsfile z_solutions_file freq_low(MHz) freq_high(MHz) ra0 dec0 tslots parallel_jobs
  import sys
  argc=len(sys.argv)
  if argc==12:
   analysis_uvwdir_loop(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],float(sys.argv[7]),float(sys.argv[8]),float(sys.argv[9]),float(sys.argv[10]),int(sys.argv[11]))
  elif argc==13:
   analysis_uvwdir_loop(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],float(sys.argv[7]),float(sys.argv[8]),float(sys.argv[9]),float(sys.argv[10]),int(sys.argv[11]),int(sys.argv[12]))
  else:
   print("Usage: python %s skymodel clusterfile uvwfile rhofile solutionsfile z_solutions_file freq_low freq_high ra0 dec0 tslots parallel_jobs"%(sys.argv[0]))
  exit()
