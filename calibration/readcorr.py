from pyrap.tables import *
import string


def read_corr(msname,outfilename):
  tt=table(msname,readonly=True)
  t1=tt.query(sortlist='TIME,ANTENNA1,ANTENNA2',columns='ANTENNA1,ANTENNA2,UVW,MODEL_DATA')
  vl=t1.getcol('MODEL_DATA')
  a1=t1.getcol('ANTENNA1')
  a2=t1.getcol('ANTENNA2')
  uvw=t1.getcol('UVW')

  
  nrtime=t1.nrows()
  
  dfile=open(outfilename,'w');
  
  for nr in range(0,nrtime):
    if (a1[nr]!=a2[nr]):
      xxd=vl[nr,0,0];
      xyd=vl[nr,0,1];
      yxd=vl[nr,0,2];
      yyd=vl[nr,0,3];
      uu=uvw[nr,0];
      vv=uvw[nr,1];
      ww=uvw[nr,2];
 
      dfile.write(str(uu)+' '+str(vv)+' '+str(ww)+' '+str(xxd.real)+' '+str(xxd.imag)+' '+ str(xyd.real)+' '+str(xyd.imag)+' '+str(yxd.real)+' '+str(yxd.imag)+' '+str(yyd.real)+' '+str(yyd.imag)+'\n')
  print(nr)
  t1.close()
  tt.close()
  


if __name__ == '__main__':
  # args MS outfile.txt (has u w v real(XX), imag(XX), same for XY, YX, YY: 11 values)
  import sys
  argc=len(sys.argv)
  if argc>2:
   read_corr(sys.argv[1],sys.argv[2])

  exit()

