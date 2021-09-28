from pyrap.tables import *

def read_corr(msname,outfilename):
  tt=table(msname,readonly=False)
  t1=tt.query(sortlist='TIME,ANTENNA1,ANTENNA2',columns='ANTENNA1,ANTENNA2,CORRECTED_DATA')
  vl=t1.getcol('CORRECTED_DATA')
  a1=t1.getcol('ANTENNA1')
  a2=t1.getcol('ANTENNA2')
  
  nrtime=t1.nrows()
  
  dfile=open(outfilename,'r');
  a=dfile.readlines()
  dfile.close()

  ct=0
  
  (nchan,_)=vl[0].shape
  for nr in range(0,nrtime):
    if (a1[nr]!=a2[nr]):
      ll=a[ct]
      ct=ct+1
      bb=ll.split();
      vl[nr,0,0]=complex(float(bb[0]),float(bb[1]));
      vl[nr,0,1]=complex(float(bb[2]),float(bb[3]));
      vl[nr,0,2]=complex(float(bb[4]),float(bb[5]));
      vl[nr,0,3]=complex(float(bb[6]),float(bb[7]));
      # also fill all other channels with same
      for ch in range(1,nchan):
        vl[nr,ch]=vl[nr,0]

 
  print(ct)
  print(nr)
  t1.putcol('CORRECTED_DATA',vl)
  t1.close()
  tt.close()
  


if __name__ == '__main__':
  # args MS outfile.txt (has re,im XX,XY,YX,YY : 8 values)
  import sys
  argc=len(sys.argv)
  if argc>2:
   read_corr(sys.argv[1],sys.argv[2])

  exit()
