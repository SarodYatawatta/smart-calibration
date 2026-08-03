from casacore.tables import *
import string


# Methods for reading, writing data in/out of a MS (without using an intermediate text file 

# read MS, given column 'colname', 
# return u,v,w, xx,xy,yx,yy (Note: excluding autocorrelations (=set to 0))
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

  return uu[:nrow],vv[:nrow],ww[:nrow],xxd[:nrow],xyd[:nrow],yxd[:nrow],yyd[:nrow]
  
# read MS
# return n_stat,uvw, obs_data,corr_data,model_Data (Note: excluding autocorrelations )
# if full_time=False, only for the first timeslot 
def read_corr_full(msname,full_time=False):
  tt=table(msname,readonly=True)
  t1=tt.query(sortlist='TIME,ANTENNA1,ANTENNA2',columns='ANTENNA1,ANTENNA2,UVW,DATA,CORRECTED_DATA,MODEL_DATA')
  data=t1.getcol('DATA')
  corr=t1.getcol('CORRECTED_DATA')
  model=t1.getcol('MODEL_DATA')
  a1=t1.getcol('ANTENNA1')
  a2=t1.getcol('ANTENNA2')
  uvw=t1.getcol('UVW')
  n_stat=len(table(msname+'/ANTENNA').getcol('NAME'))
  
  nrtime=t1.nrows()

  n_base=n_stat*(n_stat-1)//2
  assert(n_base<=nrtime)
  if not full_time:
     # only handle 1 simeslot 
     uvw_d=np.zeros((n_base,3),dtype=np.float32)
     data_d=np.zeros((n_base,8),dtype=np.float32)
     model_d=np.zeros((n_base,8),dtype=np.float32)
     corr_d=np.zeros((n_base,8),dtype=np.float32)
     max_rows=min(nrtime,n_stat*(n_stat+1)//2)
  else:
     n_time=nrtime//(n_stat*(n_stat+1)//2)
     uvw_d=np.zeros((n_base*n_time,3),dtype=np.float32)
     data_d=np.zeros((n_base*n_time,8),dtype=np.float32)
     model_d=np.zeros((n_base*n_time,8),dtype=np.float32)
     corr_d=np.zeros((n_base*n_time,8),dtype=np.float32)
     max_rows=n_time*(n_stat*(n_stat+1)//2)

  nrow=0
  for nr in range(max_rows):
    if (a1[nr]!=a2[nr]):
      data_d[nrow,0]=data[nr,0,0].real
      data_d[nrow,1]=data[nr,0,0].imag
      data_d[nrow,2]=data[nr,0,1].real
      data_d[nrow,3]=data[nr,0,1].imag
      data_d[nrow,4]=data[nr,0,2].real
      data_d[nrow,5]=data[nr,0,2].imag
      data_d[nrow,6]=data[nr,0,3].real
      data_d[nrow,7]=data[nr,0,3].imag
      corr_d[nrow,0]=corr[nr,0,0].real
      corr_d[nrow,1]=corr[nr,0,0].imag
      corr_d[nrow,2]=corr[nr,0,1].real
      corr_d[nrow,3]=corr[nr,0,1].imag
      corr_d[nrow,4]=corr[nr,0,2].real
      corr_d[nrow,5]=corr[nr,0,2].imag
      corr_d[nrow,6]=corr[nr,0,3].real
      corr_d[nrow,7]=corr[nr,0,3].imag
      model_d[nrow,0]=model[nr,0,0].real
      model_d[nrow,1]=model[nr,0,0].imag
      model_d[nrow,2]=model[nr,0,1].real
      model_d[nrow,3]=model[nr,0,1].imag
      model_d[nrow,4]=model[nr,0,2].real
      model_d[nrow,5]=model[nr,0,2].imag
      model_d[nrow,6]=model[nr,0,3].real
      model_d[nrow,7]=model[nr,0,3].imag

      uvw_d[nrow,0]=uvw[nr,0]
      uvw_d[nrow,1]=uvw[nr,1]
      uvw_d[nrow,2]=uvw[nr,2]
      nrow+=1
 
  t1.close()
  tt.close()

  return n_stat,uvw_d,data_d,corr_d,model_d
  

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


# write to MS
# input: MS name,
# column: col name
# data_d: 8*B data array, B: baselines (per timeslot)
# if full_time=False, only for the first timeslot, else full data
def write_corr_full(msname,column,data_d,full_time=False):
  tt=table(msname,readonly=False)
  t1=tt.query(sortlist='TIME,ANTENNA1,ANTENNA2',columns='ANTENNA1,ANTENNA2,'+str(column))
  data=t1.getcol(column)
  a1=t1.getcol('ANTENNA1')
  a2=t1.getcol('ANTENNA2')
  n_stat=len(table(msname+'/ANTENNA').getcol('NAME'))
  
  nrtime=t1.nrows()

  # set data to zero
  data[:,:,:]=0.0

  n_base=n_stat*(n_stat-1)//2
  # make sure data_d has correct shape
  assert(n_base<=nrtime)
  if not full_time:
     # only handle 1 simeslot 
     assert(data_d.shape[0]==8*n_base)
     data_d=data_d.reshape(n_base,8)
  else:
     n_time=nrtime//(n_stat*(n_stat+1)//2)
     assert(data_d.shape[0]==8*n_base*n_time)
     data_d=data_d.reshape(n_time*n_base,8)
  nrow=0
  for nr in range(nrtime):
    if (a1[nr]!=a2[nr]):
      data[nr,0,0]=data_d[nrow,0]+1j*data_d[nrow,1]
      data[nr,0,1]=data_d[nrow,2]+1j*data_d[nrow,3]
      data[nr,0,2]=data_d[nrow,4]+1j*data_d[nrow,5]
      data[nr,0,3]=data_d[nrow,6]+1j*data_d[nrow,7]
      nrow+=1
 
  t1.putcol(column,data)
  t1.close()
  tt.close()
