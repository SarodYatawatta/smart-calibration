#!/usr/bin/env python
from casacore.tables import *
import numpy
import string

## adds an extra column to an MS
def add_col(msname,colname):
  tt=table(msname,readonly=False)
  cl=tt.getcol('DATA')
  (nrows,nchans,npols)=cl.shape
  vl=np.zeros(shape=cl.shape,dtype='complex64')
  dmi=tt.getdminfo('DATA')
  dmi['NAME']=colname
  mkd=maketabdesc(makearrcoldesc(colname,shape=numpy.array(numpy.zeros([nchans,npols])).shape,valuetype='complex',value=0.))
  tt.addcols(mkd,dmi)
  tt.putcol(colname,vl)
  tt.close()

if __name__ == '__main__':
  # args MS COLNAME
  import sys
  argc=len(sys.argv)
  if argc>2:
   add_col(sys.argv[1],sys.argv[2])

  exit()
