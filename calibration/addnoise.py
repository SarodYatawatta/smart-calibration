#!/usr/bin/env python
import pyrap.tables as pt
import numpy as np
import string


def read_corr(msname):
  tt=pt.table(msname,readonly=False)
  c=tt.getcol('DATA')
  S=np.linalg.norm(c)
  n=(np.random.normal(-1,1,c.shape)+1j*np.random.normal(-1,1,c.shape))
  # mean should be zero
  n=n-np.mean(n)
  N=np.linalg.norm(n)
  scalefac=0.05*(S/N)
  tt.putcol('DATA',c+n*scalefac)
  tt.close()
  

if __name__ == '__main__':
  # addes noise to MS
  #args MS 
  import sys
  argc=len(sys.argv)
  if argc==2:
   read_corr(sys.argv[1])

  exit()
