from casacore.tables import *
import string


# Methods for reading, writing data in/out of a MS (without using an intermediate text file 

# read MS, given column 'colname', 
# return u,v,w, xx,xy,yx,yy (Note: excluding autocorrelations)
def read_corr(msname,colname='MODEL_DATA'):
  tt=table(msname,readonly=True)
  t1=tt.query(sortlist='TIME,ANTENNA1,ANTENNA2',columns='ANTENNA1,ANTENNA2,UVW,'+str(colname))
  vl=t1.getcol(str(colname))
  a1=t1.getcol('ANTENNA1')
  a2=t1.getcol('ANTENNA2')
  uvw=t1.getcol('UVW')

  
  nrtime=t1.nrows()
  
  uu=np.zeros(nrtime,dtype=np.float32)
  vv=np.zeros(nrtime,dtype=np.float32)
  ww=np.zeros(nrtime,dtype=np.float32)
  xxd=np.zeros(nrtime,dtype=np.csingle)
  xyd=np.zeros(nrtime,dtype=np.csingle)
  yxd=np.zeros(nrtime,dtype=np.csingle)
  yyd=np.zeros(nrtime,dtype=np.csingle)
  nrow=0
  for nr in range(0,nrtime):
    if (a1[nr]!=a2[nr]):
      xxd[nrow]=vl[nr,0,0];
      xyd[nrow]=vl[nr,0,1];
      yxd[nrow]=vl[nr,0,2];
      yyd[nrow]=vl[nr,0,3];
      uu[nrow]=uvw[nr,0];
      vv[nrow]=uvw[nr,1];
      ww[nrow]=uvw[nr,2];
      nrow+=1
 
  t1.close()
  tt.close()

  return uu,vv,ww,xxd,xyd,yxd,yyd
  

# write the correlations xx,xy,yx,yy to the MS, in column colname
# Note: autocorrelations are excluded
def write_corr(msname,xx,xy,yx,yy,colname='CORRECTED_DATA'):
  tt=table(msname,readonly=False)
  t1=tt.query(sortlist='TIME,ANTENNA1,ANTENNA2',columns='ANTENNA1,ANTENNA2,'+str(colname))
  vl=t1.getcol(colname)
  a1=t1.getcol('ANTENNA1')
  a2=t1.getcol('ANTENNA2')
  
  nrtime=t1.nrows()
  
  (nchan,_)=vl[0].shape
  nrow=0
  for nr in range(0,nrtime):
    if (a1[nr]!=a2[nr]):
      vl[nr,0,0]=xx[nrow]
      vl[nr,0,1]=xy[nrow]
      vl[nr,0,2]=yx[nrow]
      vl[nr,0,3]=yy[nrow]
      nrow+=1
      # also fill all other channels with same
      for ch in range(1,nchan):
        vl[nr,ch]=vl[nr,0]

 
  t1.putcol(colname,vl)
  t1.close()
  tt.close()
