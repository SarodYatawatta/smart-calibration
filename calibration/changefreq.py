#!/usr/bin/env python
from pyrap.tables import *


# change freq
def read_corr(msname,freq):
  import os
  import math
  tf=table(msname+'/SPECTRAL_WINDOW',readonly=False)
  ch0=tf.getcol('CHAN_FREQ')
  _,nchan=ch0.shape
  reffreq=tf.getcol('REF_FREQUENCY')
  if nchan==1:
   # single channel
   ch0[0,0]=freq
  else:
   # multi channel
   # get bandwidth
   bw=tf.getcol('EFFECTIVE_BW')
   BW=np.sum(bw)/nchan
   # lowest freq
   f0=freq-(nchan-1)*BW/2.0
   for ch in range(nchan):
     ch0[0,ch]=f0+ch*bw[0,ch]
   
  reffreq[0]=freq
  tf.putcol('CHAN_FREQ',ch0)
  tf.putcol('REF_FREQUENCY',reffreq)
  tf.close()


if __name__ == '__main__':
  # args MS
  import sys
  argc=len(sys.argv)
  if argc==3:
   read_corr(sys.argv[1],float(sys.argv[2]))
  else:
   print("thisscript MS frequency")
  exit()
