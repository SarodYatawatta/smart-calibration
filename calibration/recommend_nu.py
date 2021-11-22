#!/usr/bin/env python
import pyrap.tables as pt
import numpy as np
import string


def read_corr(msname):
  # window size to sample
  WINDOW=100
  tt=pt.table(msname,readonly=True)
  w=tt.getcol('WEIGHT_SPECTRUM')
  d=tt.getcol('DATA')
  assert(w.shape==d.shape)
  nrows,nchan,npol=w.shape
  # sample a random set of rows
  nsample=np.random.randint(0,nrows)
  # random channel
  chan=np.random.randint(0,nchan)
  nlow=max((0,nsample-WINDOW//2))
  nhigh=min((nrows-1,nsample+WINDOW//2))
  # form Stokes V
  XY=d[nlow:nhigh,chan,1]
  w=w[nlow:nhigh,chan,0]
  YX=d[nlow:nhigh,chan,2]
  VV=XY-YX
  # sqr
  VV2=VV**2
  # multiply by weight
  wV=w*VV2
  ratio=np.sum(np.abs(wV))/WINDOW
  print('residual^2 x weight_spectrum ~ %e'%(ratio))
  print('nu=2 is optimal for ~ 0.001, so optimal nu range %d:%d'%(int(2*ratio/0.001),int(20*ratio/0.001)))
  print('(run this script a couple of times to get a good sampling of data)')
  tt.close()


if __name__ == '__main__':
  # args: MS 
  # Recommend optimal value of robust nu to use based on 
  # the residual^2 and weight spectrum of the data in the MS
  import sys
  argc=len(sys.argv)
  if argc==2:
   read_corr(sys.argv[1])

  exit()

