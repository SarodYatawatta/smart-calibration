import torch
import numpy as np
import math

def radectolm(ra,dec,ra0,dec0):
# return source direction cosines [l,m,n] obtained for a source at spherical 
# coordinates (ra,dec) with respect to phase center (ra0,dec0).
  if dec0<0.0 and dec>=0.0:
    dec0=dec0+2.0*math.pi

  l=math.sin(ra-ra0)*math.cos(dec)
  m=-(math.cos(ra-ra0)*math.cos(dec)*math.sin(dec0)-math.cos(dec0)*math.sin(dec))
  n=(math.sqrt(1.-l*l-m*m)-1.)

  return (l,m,n)


def lmtoradec(l,m,ra0,dec0):
  sind0=math.sin(dec0)
  cosd0=math.cos(dec0)
  dl=l
  dm=m
  d0=pow(dm,2)*pow(sind0,2)+pow(dl,2)-2*dm*cosd0*sind0
  sind=math.sqrt(abs(pow(sind0,2)-d0))
  cosd=math.sqrt(abs(pow(cosd0,2)+d0))
  if sind0>0:
   sind=abs(sind)
  else:
   sind=-abs(sind)

  dec=math.atan2(sind,cosd)
  if l != 0:
   ra=math.atan2(-dl,cosd0-dm*sind0)+ra0
  else:
   ra=atan2(1e-10,cosd0-dm*sind0)+ra0

  return ra,dec


def radToRA(rad):
# Radians to RA=[hr,min,sec]
# Rad=(hr+min/60+sec/60*60)*pi/12
  # convert negative values
  if rad <0:
   rad=rad+2*math.pi

  tmpval=rad*12.0/math.pi
  hr=math.floor(tmpval)
  tmpval=tmpval-hr
  tmpval=tmpval*60
  mins=math.floor(tmpval)
  tmpval=tmpval-mins
  tmpval=tmpval*60
  sec=tmpval
  hr=hr%24
  mins=mins%60

  return hr,mins,sec


def radToDec(rad):
# Radians to Dec=[hr,min,sec]
# Rad=(hr+min/60+sec/60*60)*pi/180
  if rad<0:
   mult=-1
   rad=abs(rad)
  else:
   mult=1

  tmpval=rad*180.0/math.pi
  hr=math.floor(tmpval)
  tmpval=tmpval-hr
  tmpval=tmpval*60
  mins=math.floor(tmpval)
  tmpval=tmpval-mins
  tmpval=tmpval*60
  sec=tmpval
  hr=mult*(hr%180)
  mins=mins%60

  return hr,mins,sec

# read solutions file, return solutions tensor and frequency
# return freq,J
def readsolutions(filename):
  fh=open(filename,'r')
  # skip first 2 lines
  next(fh)
  next(fh)
  # freq/MHz BW/MHz time/min N K Ktrue
  curline=next(fh)
  cl=curline.split()
  freq=float(cl[0])*1e6
  Ns=int(cl[3]) # stations
  K=int(cl[5]) # true directions
  fullset=fh.readlines()
  fh.close()
  Nt=len(fullset)
  Nto=Nt//(8*Ns)
  a=np.zeros((Nt,K),dtype=np.float32)
  ci=0
  for cl in fullset:
    cl1=cl.split()
    for cj in range(len(cl1)-1):
      a[ci,cj]=float(cl1[cj+1])
    ci +=1
  
  J=np.zeros((K,2*Ns*Nto,2),dtype=np.csingle)
  for m in range(K):
    for n in range(Ns):
      J[m,2*n:2*Ns*Nto:2*Ns,0]=a[8*n:Nto*8*Ns:Ns*8,m]+1j*a[8*n+1:Nto*8*Ns:Ns*8,m]
      J[m,2*n:2*Ns*Nto:2*Ns,1]=a[8*n+2:Nto*8*Ns:Ns*8,m]+1j*a[8*n+3:Nto*8*Ns:Ns*8,m]
      J[m,2*n+1:2*Ns*Nto:2*Ns,0]=a[8*n+4:Nto*8*Ns:Ns*8,m]+1j*a[8*n+5:Nto*8*Ns:Ns*8,m]
      J[m,2*n+1:2*Ns*Nto:2*Ns,1]=a[8*n+6:Nto*8*Ns:Ns*8,m]+1j*a[8*n+7:Nto*8*Ns:Ns*8,m]

  return (freq,J)


# read solutions file for spatial model, return solutions Zspat tensor 
# return N(stations), F(freq poly) theta,phi (polar coordinate of calibration dirs Kx1) and Z
def read_spatial_solutions(filename):
  fh=open(filename,'r')
  # skip first 3 lines
  next(fh)
  next(fh)
  next(fh)
  # reference_freq/MHz Freqpol(F) Spatialpol(G) N K Ktrue
  # F, G are number of polynomials in frequency and space
  curline=next(fh)
  cl=curline.split()
  freq=float(cl[0])*1e6
  F=int(cl[1]) # polynomials in frequency
  G=int(cl[2]) # polynomials in space (spherical harmonics)
  Ns=int(cl[3]) # stations
  K=int(cl[5]) # true directions
  # spherical harmonic order
  n0=int(math.sqrt(G))
  # next two lines, Kx1 values for source coordinates
  curline=next(fh)
  cl=curline.split()
  thetak=[float(x) for x in cl]
  curline=next(fh)
  cl=curline.split()
  phik=[float(x) for x in cl]
  assert(len(phik)==len(thetak) and len(phik)==K)
  # the remaining lines will have 1+G columns, first col from 0..8FN-1
  fullset=fh.readlines()
  fh.close()
  Nt=len(fullset)
  Nto=Nt//(8*F*Ns)
  a=np.zeros((Nt,G),dtype=np.float32)
  ci=0
  for cl in fullset:
    cl1=cl.split()
    for cj in range(len(cl1)-1):
      a[ci,cj]=float(cl1[cj+1])
    ci +=1

  Z=np.zeros((Nto,2*F*Ns,2*G),dtype=np.csingle)
  for ci in range(Nto):
    # split each col of a[] to 4FN x 2, 
    # each 4FN yields 2FN complex, which is one col of Z
    for cj in range(G):
      b=a[(8*F*Ns)*ci:(8*F*Ns)*(ci+1),cj]
      c=b[0:8*F*Ns:2]+1j*b[1:8*F*Ns:2]
      Z[ci,:,2*cj]=c[0:2*F*Ns]
      Z[ci,:,2*cj+1]=c[2*F*Ns:4*F*Ns]
  return Ns,F,thetak,phik,Z


# return K,C
def skytocoherencies(skymodel,clusterfile,uvwfile,N,freq,ra0,dec0):
# use skymodel,clusterfile and predict coherencies for uvwfile coordinates
# C: K way tensor, each slice Tx4, T: total samples, 4: XX,XY(=0),YX(=0),YY
# N : stations, ra0,dec0: phase center (rad), freq: frequency
  # light speed
  c=2.99792458e8
  
  # uvw file
  fh=open(uvwfile,'r')
  fullset=fh.readlines()
  fh.close()
  # total samples=baselines x timeslots
  T=len(fullset)
  uu=np.zeros(T,dtype=np.float32)
  vv=np.zeros(T,dtype=np.float32)
  ww=np.zeros(T,dtype=np.float32)
  ci=0
  for cl in fullset:
   cl1=cl.split()
   uu[ci]=float(cl1[0])
   vv[ci]=float(cl1[1])
   ww[ci]=float(cl1[2])
   ci +=1

  uu *=2.0*math.pi/c*freq
  vv *=2.0*math.pi/c*freq
  ww *=2.0*math.pi/c*freq
  del fullset

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
  # coherencies: K clusters, T rows, 4=XX,XY,YX,YY
  C=np.zeros((K,T,4),dtype=np.csingle)

  # output sky/cluster info for input to DQN
  # format of each line: cluster_id l m sI sP
  #fh=open('./skylmn.txt','w+')
  ck=0 # cluster id
  for cl in fullset:
   if (not cl.startswith('#')) and len(cl)>1:
     cl1=cl.split()
     for sname in cl1[2:]:
       # 3:ra 3:dec sI 0 0 0 sP 0 0 0 0 0 0 freq0
       sinfo=S[sname]
       mra=(float(sinfo[0])+float(sinfo[1])/60.+float(sinfo[2])/3600.)*360./24.*math.pi/180.0
       mdec=(float(sinfo[3])+float(sinfo[4])/60.+float(sinfo[5])/3600.)*math.pi/180.0
       (myll,mymm,mynn)=radectolm(mra,mdec,ra0,dec0)
       mysI=float(sinfo[6])
       f0=float(sinfo[17])
       fratio=math.log(freq/f0)
       sIo=math.exp(math.log(mysI)+float(sinfo[10])*fratio+float(sinfo[11])*math.pow(fratio,2)+float(sinfo[12])*math.pow(fratio,3))
       # add to C
       uvw=(uu*myll+vv*mymm+ww*mynn)
       XX=(np.cos(uvw)+1j*np.sin(uvw))*sIo
       C[ck,:,0]=C[ck,:,0]+XX
       #fh.write(str(ck)+' '+str(myll)+' '+str(mymm)+' '+str(mysI)+' '+str(sinfo[10])+'\n')
     ck+=1
  #fh.close()

  # copy to YY 
  for ck in range(K):
    C[ck,:,3]=C[ck,:,0]

  return K,C
  
# return K,C
def skytocoherencies_uvw(skymodel,clusterfile,uu,vv,ww,N,freq,ra0,dec0):
# use skymodel,clusterfile and predict coherencies for uvwfile coordinates
# C: K way tensor, each slice Tx4, T: total samples, 4: XX,XY(=0),YX(=0),YY
# N : stations, ra0,dec0: phase center (rad), freq: frequency
  # light speed
  c=2.99792458e8
  
  uu *=2.0*math.pi/c*freq
  vv *=2.0*math.pi/c*freq
  ww *=2.0*math.pi/c*freq

  # bandwidth for smear factor
  fdelta=180e3/freq

  # total number of samples
  T=len(uu)
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
  # coherencies: K clusters, T rows, 4=XX,XY,YX,YY
  C=np.zeros((K,T,4),dtype=np.csingle)

  # output sky/cluster info for input to DQN
  # format of each line: cluster_id l m sI sP
  #fh=open('./skylmn.txt','w+')
  ck=0 # cluster id
  for cl in fullset:
   if (not cl.startswith('#')) and len(cl)>1:
     cl1=cl.split()
     for sname in cl1[2:]:
       # 3:ra 3:dec sI 0 0 0 sP 0 0 0 0 0 0 freq0
       sinfo=S[sname]
       # parse first char to see if this is a GAUSSIAN
       gaussian_src=False
       if sname[0]=='G':
           gaussian_src=True
       mra=(float(sinfo[0])+float(sinfo[1])/60.+float(sinfo[2])/3600.)*360./24.*math.pi/180.0
       mdec=(float(sinfo[3])+float(sinfo[4])/60.+float(sinfo[5])/3600.)*math.pi/180.0
       (myll,mymm,mynn)=radectolm(mra,mdec,ra0,dec0)
       mysI=float(sinfo[6])
       f0=float(sinfo[17])
       fratio=math.log(freq/f0)
       sIo=math.exp(math.log(mysI)+float(sinfo[10])*fratio+float(sinfo[11])*math.pow(fratio,2)+float(sinfo[12])*math.pow(fratio,3))
       uvw=(uu*myll+vv*mymm+ww*mynn)
       # smear factor, use normalization for numpy sinc() version
       smearfactor=np.abs(np.sinc(uvw*0.5*fdelta/np.pi))
       # add to C
       if gaussian_src:
         # projection parameters
         phi=-math.acos(mynn)
         xi=-math.atan2(-myll,mymm)
         cxi=math.cos(xi)
         sxi=math.sin(xi)
         cphi=math.cos(phi)
         sphi=math.sin(phi)
         eP=float(sinfo[16])
         eX=2*float(sinfo[14])
         eY=2*float(sinfo[15])
         uup=uu*cxi-vv*cphi*sxi+ww*sphi*sxi
         vvp=uu*sxi+vv*cphi*cxi-ww*sphi*cxi
         cpa=math.cos(eP)
         spa=math.sin(eP)
         uut=eX*(cpa*uup-spa*vvp)
         vvt=eY*(spa*uup+cpa*vvp)
         scalefac=0.5*math.pi*np.exp(-(uut*uut+vvt*vvt))
         XX=(np.cos(uvw)+1j*np.sin(uvw))*sIo*scalefac*smearfactor
       else:
         XX=(np.cos(uvw)+1j*np.sin(uvw))*sIo*smearfactor
       C[ck,:,0]=C[ck,:,0]+XX
       #fh.write(str(ck)+' '+str(myll)+' '+str(mymm)+' '+str(mysI)+' '+str(sinfo[10])+'\n')
     ck+=1
  #fh.close()

  # copy to YY 
  for ck in range(K):
    C[ck,:,3]=C[ck,:,0]

  return K,C


# return rho Kx1 vector
def read_rho(rhofile,K):
# initialize rho from text file
  ci=0
  rho=np.zeros(K,dtype=np.float32)
  with open(rhofile,'r') as fh:
    for curline in fh:
      if (not curline.startswith('#')) and len(curline)>1:
         curline1=curline.split()
         # id hybrid rho
         rho[ci]=float(curline1[2])
         ci +=1

  return rho


# return skymodel reading M components, M>2
def read_skycluster(skyclusterfile,M):
# sky/cluster model text file
# format: cluster_id l m sI sP
  ci=0
  skl=np.zeros((M,5),dtype=np.float32)
  with open(skyclusterfile,'r') as fh:
    for curline in fh:
      if (not curline.startswith('#')) and len(curline)>1:
         curline1=curline.split()
         # cluster_id l m sI sP
         for cj in range(5):
          skl[ci,cj]=float(curline1[cj])
         ci +=1

  return skl

# return XX,XY,YX,YY :each Tx1 complex vectors
def readuvw(uvwfile):
  a=np.loadtxt(uvwfile,delimiter=' ')
  # read u,v,w,xx(re,im), xy(re,im) yx(re,im) yy(re,im)
  XX=a[:,3]+1j*a[:,4]
  XY=a[:,5]+1j*a[:,6]
  YX=a[:,7]+1j*a[:,8]
  YY=a[:,9]+1j*a[:,10]
  return XX,XY,YX,YY

# write XX,XY,YX,YY to text file
def writeuvw(uvwfile,XX,XY,YX,YY):
  # collect to a tuple
  dfile=open(uvwfile,'w+')
  T=XX.shape[0]
  for ci in range(T):
   xxyy=str(XX[ci].real)+' '+str(XX[ci].imag)+' '+str(XY[ci].real)+' '+str(XY[ci].imag)+' '+str(YX[ci].real)+' '+str(YX[ci].imag)+' '+str(YY[ci].real)+' '+str(YY[ci].imag)+'\n'
   dfile.write(xxyy)
  dfile.close()

def Bpoly(x,N):
# evaluate Bernstein basis functions
# x a vector of values in [0,1]
# y : for each x, N+1 values of the Bernstein basis evaluated at x
# [N_C_0 x^0 (1-x)^(N-0) , N_C_1 x^1 (1-x)^(N-1), ..., N_C_r x^r (1-x)^(N-r), ... , N_C_N x^N (1-x)^0 ]
# N_C_r = N!/(N-r)!r!
 M=len(x)
 # need array of factorials [0!,1!,...,N!]
 fact=np.ones(N+1,dtype=np.float32)
 for ci in range(1,N+1):
  fact[ci]=fact[ci-1]*(ci)
 # need powers of x and (1-x)
 px=np.ones((N+1,M),dtype=np.float32)
 p1x=np.ones((N+1,M),dtype=np.float32)
 for ci in range(1,N+1):
   px[ci,:]=px[ci-1,:]*x
   p1x[ci,:]=p1x[ci-1,:]*(1.-x)

 y=np.zeros((N+1,M),dtype=np.float32)
 for ci in range(1,N+2): # r goes from 0 to N
   # N_C_r x^r (1-x)^(N-r)
   y[ci-1]=fact[N]/(fact[N-ci+1]*fact[ci-1])*px[ci-1]*p1x[N-ci+1]

 return y.transpose()
  
 
# return F: 2Nx2N, and P: 2N Ne x 2 N
def consensus_poly(Ne,N,freqs,f0,fidx,polytype=0,rho=0.0,alpha=0.0):
 # Ne: polynomial order (number of terms)
 # N: stations
 # freqs: Nfx1 freq vector
 # f0: reference freq
 # fidx: 0,1,... frequency index to create F (working frequency)
 # polytype:0 ordinary, 1 Bernstein 
 # alpha: regularization parameter in federated averaging/spatial constraints
 Nf=len(freqs)
 Bfull=np.zeros((Nf,Ne),dtype=np.float32)
 if (polytype==0):
  Bfull[:,0]=1.
  ff=(freqs-f0)/f0
  for cj in range(1,Ne):
    Bfull[:,cj]=np.power(ff,cj)
 else:
  ff=(freqs-freqs.min())/(freqs.max()-freqs.min())
  Bfull=Bpoly(ff,Ne-1)

 Bi=np.zeros((Ne,Ne),dtype=np.float32)
 for cf in range(Nf):
   Bi=Bi+np.outer(Bfull[cf],Bfull[cf])

 # federated averaging/spatial constraing comes in as alpha x I
 Bi=np.linalg.pinv(Bi+alpha*np.eye(Ne))
 # select correct freq. component
 Bf=np.kron(Bfull[fidx],np.eye(2*N))
 P=np.matmul(np.kron(Bi,np.eye(2*N)),Bf.transpose())
 F=np.eye(2*N)-rho*np.matmul(Bf,P)

 return F,P
  


# return H=K x 4Nx4N tensor
def Hessianres(R,C,J,N):
# B: baselines=N(N-1)/2
# T: timeslots for this interval
# R: 2*B*Tx2 - residual for this interval
# C: KxB*Tx4 - coherencies for this interval
# J: Kx2Nx2 - valid solution for this interval

# instead of using input V, use residual R to calculate the Hessian
# Hess is 4Nx4N matrix, build by accumulating 4x4 kron products into a NxN block matrix
# and averaging over T time slots
# notation:
# Y \kron A_p^T ( Z ) A_q means p-th row, q-th col block is replaced by Y \kron Z (4x4) matrix
# res_pq= V_pq - J_p C_pq J_q^H
# then, p,q baseline contribution 
# -C^\star kron A_p^T res  A_q - C^T kron A_q^T res^H A_p
# + (C J_q^H J_q C^H)^T kron A_p^T A_p + (C^H J_p^H J_p C)^T kron A_q^T A_q

  B=N*(N-1)//2
  T=R.shape[0]//(2*B)
  K=C.shape[0]
  
  H=np.zeros((K,4*N,4*N),dtype=np.csingle)

  for k in range(K):
    ck=0 
    for cn in range(T):
       for p in range(N-1):
          for q in range(p+1,N):
             Res=R[2*ck:2*(ck+1),:]
             Ci=C[k,ck,:].reshape((2,2),order='F')
             Imp=np.kron(-np.conj(Ci),Res)
             H[k,4*p:4*(p+1),4*q:4*(q+1)] +=Imp
             H[k,4*q:4*(q+1),4*p:4*(p+1)] +=np.conj(Imp.transpose())
             Res1=np.matmul(Ci,np.conj(J[k,2*q:2*(q+1),:].transpose()))
             Res=np.matmul(Res1,np.conj(Res1.transpose()))
             H[k,4*p:4*(p+1),4*p:4*(p+1)] +=np.kron(Res.transpose(),np.eye(2))
             Res1=np.matmul(J[k,2*p:2*(p+1),:],Ci)
             Res=np.matmul(np.conj(Res1.transpose()),Res1)
             H[k,4*q:4*(q+1),4*q:4*(q+1)] +=np.kron(Res.transpose(),np.eye(2))
             ck+=1
             del Res,Res1,Imp,Ci
  return H/(B*T)


# return dJ=K x 4N x 4B tensor
def Dsolutions(C,J,N,Dgrad,r):
# B: baselines=N(N-1)/2
# T: timeslots for this interval
# BT: BxT
# C: KxB*Tx4 - coherencies for this interval
# J: Kx2Nx2 - valid solution for this interval
# evaluate vec(\partial J/ \partial x_pp,qq) for all possible pp,qq (baselines)
# Dgrad is K x 4Nx4N tensor, build by accumulating 4x4 kron products into a NxN block matrix
# r: 0,1,2...7 : determine which element of 2x2 matrix is 1
# return dJ : K x 4N x N(N-1)/2 matrix (note: for each baseline, the values are averaged over timeslots)
  B=N*(N-1)//2
  T=C.shape[1]//B
  K=C.shape[0]

  dJ=np.zeros((K,4*N,B),dtype=np.csingle)

  EPS=1e-12 # overcome singular matrix
  # setup 4x1 vector, one goes to depending on r
  rr=np.zeros(8,dtype=np.float32)
  rr[r]=1.
  dVpq=rr[0:8:2]+1j*rr[1:8:2]

  for k in range(K):
    # ck will fill each column
    ck=0
    # setup 4N x B matrix (fixme: use a sparse matrix)
    AdV=np.zeros((4*N,B),dtype=np.csingle)
    for cn in range(T):
      for p in range(N-1):
         for q in range(p+1,N):
            # fill up column ck of AdV 
            # left hand side (J_q C^H)^T , right hand side I
            # kron product will fill only rows 4*(p-1)+1:4*p
            Ci=C[k,ck,:].reshape((2,2),order='F')
            lhs=np.matmul(J[k,2*q:2*(q+1),:],np.conj(Ci.transpose()))
            fillvex=np.matmul(np.kron(lhs.transpose(),np.eye(2)),dVpq)
            AdV[4*p:4*(p+1),ck%B] +=fillvex
            ck +=1
    
    dJ[k]=np.linalg.solve(Dgrad[k]+EPS*np.eye(4*N),AdV)


  return dJ

# return dJ= 8 x K x 4N x 4B tensor (for all possible r values)
def Dsolutions_r(C,J,N,Dgrad):
# B: baselines=N(N-1)/2
# T: timeslots for this interval
# BT: BxT
# C: KxB*Tx4 - coherencies for this interval
# J: Kx2Nx2 - valid solution for this interval
# evaluate vec(\partial J/ \partial x_pp,qq) for all possible pp,qq (baselines)
# Dgrad is K x 4Nx4N tensor, build by accumulating 4x4 kron products into a NxN block matrix
# loop over r: 0,1,2...7 : determine which element of 2x2 matrix is 1 (first dimension in dJ)
# return dJ : 8 x K x 4N x N(N-1)/2 matrix (note: for each baseline, the values are averaged over timeslots)
  B=N*(N-1)//2
  T=C.shape[1]//B
  K=C.shape[0]

  dJ=np.zeros((8,K,4*N,B),dtype=np.csingle)

  EPS=1e-12 # overcome singular matrix
  for k in range(K):
    # ck will fill each column
    ck=0
    # setup 4N x B matrix (fixme: use a sparse matrix)
    AdV=np.zeros((8,4*N,B),dtype=np.csingle)
    for cn in range(T):
      for p in range(N-1):
         for q in range(p+1,N):
            # fill up column ck of AdV 
            # left hand side (J_q C^H)^T , right hand side I
            # kron product will fill only rows 4*(p-1)+1:4*p
            Ci=C[k,ck,:].reshape((2,2),order='F')
            lhs=np.matmul(J[k,2*q:2*(q+1),:],np.conj(Ci.transpose()))
            for r in range(8):
              # setup 4x1 vector, one goes to depending on r
              rr=np.zeros(8,dtype=np.float32)
              rr[r]=1.
              dVpq=rr[0:8:2]+1j*rr[1:8:2]
              fillvex=np.matmul(np.kron(lhs.transpose(),np.eye(2)),dVpq)
              AdV[r,4*p:4*(p+1),ck%B] +=fillvex
            ck +=1
    
    # iterate over r
    dJ[0:8,k]=np.linalg.solve(Dgrad[k]+EPS*np.eye(4*N),AdV[0:8])


  return dJ



# dR: 4B x B (sum up all K)
def Dresiduals(C,J,N,dJ,addself,r):
# B: baselines=N(N-1)/2
# T: timeslots for this interval
# BT: BxT
# C: KxB*Tx4 - coherencies for this interval
# J: Kx2Nx2 - valid solution for this interval
# dJ: Kx4Nx4B
# to find vec(\partial V_pq / \partial x_pp,qq) - vec(\partial (eq 24)_pq / \partial x_pp,qq), select rows 4*(p-1)+1:4p from dSol
# note: dJ : K x 4N x N(N-1)/2 for all possible pp,qq combinations
# dR: 4B x B, rows for all possible p,q and columns for all possible pp,qq (averaged over all time slots)
# p,q: B rows (4 times) : 4B rows,
# columns : B for all possible pp,qq values
# note: r 1,2...8 : determine which element of 2x2 matrix (\partialV_pq/\partial x_pp,qq) is 1
# r should be the same value used to find dSol
# addself: if 1, add (\partialV_pq/\partial x_pp,qq) to the block diagonal
  B=N*(N-1)//2
  T=C.shape[1]//B
  K=C.shape[0]

  dR=np.zeros((4*B,B),dtype=np.csingle)
  # setup 4x1 vector, one goes to depending on r
  rr=np.zeros(8,dtype=np.float32)
  rr[r]=1.
  dVpq=rr[0:8:2]+1j*rr[1:8:2]
  for k in range(K):
    # ck will fill each column
    ck=0
    for cn in range(T):
      for p in range(N-1):
         for q in range(p+1,N):
            # fill up ck-th block of 4 rows in dR
            # left hand side -(C J_q^H)^T , right hand side I
            Ci=C[k,ck,:].reshape((2,2),order='F')
            lhs=-np.matmul(Ci,np.conj(J[k,2*q:2*(q+1),:].transpose())).transpose()
            # kron product will fill only rows 4*(p-1)+1:4*p, column ck of dJ
            rhs=dJ[k,4*p:4*(p+1),:]
            fillvex=np.matmul(np.kron(lhs,np.eye(2)),rhs)
            ck1=ck%B
            if addself:
              fillvex[:,ck1] +=dVpq

            dR[4*ck1:4*(ck1+1),:] +=fillvex
            ck +=1

  return dR/(B*T)

# dR: K x 4B x B (for all K)
def Dresiduals_k(C,J,N,dJ,addself,r):
# B: baselines=N(N-1)/2
# T: timeslots for this interval
# BT: BxT
# C: KxB*Tx4 - coherencies for this interval
# J: Kx2Nx2 - valid solution for this interval
# dJ: Kx4Nx4B
# to find vec(\partial V_pq / \partial x_pp,qq) - vec(\partial (eq 24)_pq / \partial x_pp,qq), select rows 4*(p-1)+1:4p from dSol
# note: dJ : K x 4N x N(N-1)/2 for all possible pp,qq combinations
# dR: 4B x B, rows for all possible p,q and columns for all possible pp,qq (averaged over all time slots)
# p,q: B rows (4 times) : 4B rows,
# columns : B for all possible pp,qq values
# note: r 1,2...8 : determine which element of 2x2 matrix (\partialV_pq/\partial x_pp,qq) is 1
# r should be the same value used to find dSol
# addself: if 1, add (\partialV_pq/\partial x_pp,qq) to the block diagonal
  B=N*(N-1)//2
  T=C.shape[1]//B
  K=C.shape[0]

  dR=np.zeros((K,4*B,B),dtype=np.csingle)
  # setup 4x1 vector, one goes to depending on r
  rr=np.zeros(8,dtype=np.float32)
  rr[r]=1.
  dVpq=rr[0:8:2]+1j*rr[1:8:2]
  for k in range(K):
    # ck will fill each column
    ck=0
    for cn in range(T):
      for p in range(N-1):
         for q in range(p+1,N):
            # fill up ck-th block of 4 rows in dR
            # left hand side -(C J_q^H)^T , right hand side I
            Ci=C[k,ck,:].reshape((2,2),order='F')
            lhs=-np.matmul(Ci,np.conj(J[k,2*q:2*(q+1),:].transpose())).transpose()
            # kron product will fill only rows 4*(p-1)+1:4*p, column ck of dJ
            rhs=dJ[k,4*p:4*(p+1),:]
            fillvex=np.matmul(np.kron(lhs,np.eye(2)),rhs)
            ck1=ck%B
            if addself:
              fillvex[:,ck1] +=dVpq

            dR[k,4*ck1:4*(ck1+1),:] +=fillvex
            ck +=1

  return dR/(B*T)


# dR: 8 x 4B x B (sum up all K) (for all possible r values)
def Dresiduals_r(C,J,N,dJ,addself):
# B: baselines=N(N-1)/2
# T: timeslots for this interval
# BT: BxT
# C: KxB*Tx4 - coherencies for this interval
# J: Kx2Nx2 - valid solution for this interval
# dJ: 8 x Kx4Nx4B (for all possible r)
# to find vec(\partial V_pq / \partial x_pp,qq) - vec(\partial (eq 24)_pq / \partial x_pp,qq), select rows 4*(p-1)+1:4p from dSol
# note: dJ : K x 4N x N(N-1)/2 for all possible pp,qq combinations
# dR: 4B x B, rows for all possible p,q and columns for all possible pp,qq (averaged over all time slots)
# p,q: B rows (4 times) : 4B rows,
# columns : B for all possible pp,qq values
# note: loop over r 1,2...8 : determine which element of 2x2 matrix (\partialV_pq/\partial x_pp,qq) is 1
# r should be the same value used to find dSol
# addself: if 1, add (\partialV_pq/\partial x_pp,qq) to the block diagonal
  B=N*(N-1)//2
  T=C.shape[1]//B
  K=C.shape[0]

  dR=np.zeros((8,4*B,B),dtype=np.csingle)
  for k in range(K):
    # ck will fill each column
    ck=0
    for cn in range(T):
      for p in range(N-1):
         for q in range(p+1,N):
            # fill up ck-th block of 4 rows in dR
            # left hand side -(C J_q^H)^T , right hand side I
            Ci=C[k,ck,:].reshape((2,2),order='F')
            lhs=-np.matmul(Ci,np.conj(J[k,2*q:2*(q+1),:].transpose())).transpose()
            # kron product will fill only rows 4*(p-1)+1:4*p, column ck of dJ
            for r in range(8):
              rhs=dJ[r,k,4*p:4*(p+1),:]

              fillvex=np.matmul(np.kron(lhs,np.eye(2)),rhs)
              ck1=ck%B
              if addself:
                # setup 4x1 vector, one goes to depending on r
                rr=np.zeros(8,dtype=np.float32)
                rr[r]=1.
                dVpq=rr[0:8:2]+1j*rr[1:8:2]
                fillvex[:,ck1] +=dVpq

              dR[r,4*ck1:4*(ck1+1),:] +=fillvex
            ck +=1

  return dR/(B*T)

# dR: 8 x K x 4B x B (for all possible r values and K)
def Dresiduals_rk(C,J,N,dJ,addself):
# B: baselines=N(N-1)/2
# T: timeslots for this interval
# BT: BxT
# C: KxB*Tx4 - coherencies for this interval
# J: Kx2Nx2 - valid solution for this interval
# dJ: 8 x Kx4Nx4B (for all possible r)
# to find vec(\partial V_pq / \partial x_pp,qq) - vec(\partial (eq 24)_pq / \partial x_pp,qq), select rows 4*(p-1)+1:4p from dSol
# note: dJ : K x 4N x N(N-1)/2 for all possible pp,qq combinations
# dR: 4B x B, rows for all possible p,q and columns for all possible pp,qq (averaged over all time slots)
# p,q: B rows (4 times) : 4B rows,
# columns : B for all possible pp,qq values
# note: loop over r 1,2...8 : determine which element of 2x2 matrix (\partialV_pq/\partial x_pp,qq) is 1
# r should be the same value used to find dSol
# addself: if 1, add (\partialV_pq/\partial x_pp,qq) to the block diagonal
  B=N*(N-1)//2
  T=C.shape[1]//B
  K=C.shape[0]

  dR=np.zeros((8,K,4*B,B),dtype=np.csingle)
  for k in range(K):
    # ck will fill each column
    ck=0
    for cn in range(T):
      for p in range(N-1):
         for q in range(p+1,N):
            # fill up ck-th block of 4 rows in dR
            # left hand side -(C J_q^H)^T , right hand side I
            Ci=C[k,ck,:].reshape((2,2),order='F')
            lhs=-np.matmul(Ci,np.conj(J[k,2*q:2*(q+1),:].transpose())).transpose()
            # kron product will fill only rows 4*(p-1)+1:4*p, column ck of dJ
            for r in range(8):
              rhs=dJ[r,k,4*p:4*(p+1),:]

              fillvex=np.matmul(np.kron(lhs,np.eye(2)),rhs)
              ck1=ck%B
              if addself:
                # setup 4x1 vector, one goes to depending on r
                rr=np.zeros(8,dtype=np.float32)
                rr[r]=1.
                dVpq=rr[0:8:2]+1j*rr[1:8:2]
                fillvex[:,ck1] +=dVpq

              dR[r,k,4*ck1:4*(ck1+1),:] +=fillvex
            ck +=1

  return dR/(B*T)


# return LLR=K x B*T tensor
# Calculate log likelihood ratio
def log_likelihood_ratio(R,C,J,N):
# B: baselines=N(N-1)/2
# T: timeslots for this interval
# R: 2*B*Tx2 - residual for this interval
# C: KxB*Tx4 - coherencies for this interval
# J: Kx2Nx2 - valid solution for this interval

# Let x: vectorized 4x1 data of each baseline, x=residual+model
# H0: x ~ CN(0,sigma^2 I), H1: x ~ CN(vec(Jp C_pq J_q^H), sigma^2 I)
# f_H1(x|mu)/ f_H0(x) ~ exp(-(x-mu)^H(1/sigma^2 I) (x-mu) + x^H (1/sigma^2 I) x)
# ~ 1/sigma^2 ( -r^H r + (r+mu)^H (r+mu) ), where r+mu=x
# sigma^2 : estimate using Stokes V
# ~ 1/sigma^2 (r^H mu+mu^H r)
# LLR: larger indicate more likely, or strong presence of source
# Note: since LLR is dependent on C, it is dependent on source_direction+freq+uv cov

  B=N*(N-1)//2
  T=R.shape[0]//(2*B)
  K=C.shape[0]
 
  LLR=np.zeros((K),dtype=np.float32)

  EPS=1e-12 # overcome division by zero
  for k in range(K):
    ck=0
    sigma2=0
    r=np.zeros((B*T*4),dtype=np.csingle)
    mu=np.zeros((B*T*4),dtype=np.csingle)
    for cn in range(T):
       for p in range(N-1):
          for q in range(p+1,N):
             Res=R[2*ck:2*(ck+1),:]
             sV=0.5*(Res[0,1]-Res[1,0])
             sigma2+=np.real(sV*np.conj(sV)+EPS)
             Ci=C[k,ck,:].reshape((2,2),order='F')
             Model=np.matmul(J[k,2*p:2*(p+1),:],np.matmul(Ci,np.conj(J[k,2*q:2*(q+1),:].transpose())))
             r[4*ck:4*(ck+1)]=Res.ravel()
             mu[4*ck:4*(ck+1)]=Model.ravel()
             ck+=1
             del Res,Ci,Model
    LLR[k]=(-np.linalg.norm(r)**2+np.linalg.norm(r+mu)**2)
    LLR[k] /= sigma2
  return LLR



#readsolutions('L_SB1.MS.solutions')
#print(radectolm(1,0.2,0.4,0.3))
#skytocoherencies('sky.txt','cluster.txt','smalluvw.txt',62,150e6,1,1.5)
#read_rho('admm_rho.txt',2)
#readuvw('smalluvw.txt')
#read_spatial_solutions('spatial_zsol')
