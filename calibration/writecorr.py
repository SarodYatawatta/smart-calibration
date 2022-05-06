from pyrap.tables import *

def read_corr(msname,outfilename,colname='CORRECTED_DATA'):
  tt=table(msname,readonly=False)
  t1=tt.query(sortlist='TIME,ANTENNA1,ANTENNA2',columns='ANTENNA1,ANTENNA2,'+str(colname))
  vl=t1.getcol(colname)
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
  t1.putcol(colname,vl)
  t1.close()
  tt.close()
  


if __name__ == '__main__':
  # args MS outfile.txt (has re,im XX,XY,YX,YY : 8 values)
  # or args MS outfile.txt columnname
  import sys
  argc=len(sys.argv)
  if argc==3:
   read_corr(sys.argv[1],sys.argv[2])
  elif argc==4:
   read_corr(sys.argv[1],sys.argv[2],sys.argv[3])

  exit()
