#!/usr/bin/env python
import math

def read_shap(oldfile,newfile):
  ff=open(oldfile,'r')
  gg=open(newfile,'w+')

  # read/write first line
  curline=next(ff)
  gg.write(curline)
  # get number of modes
  curline=next(ff)
  gg.write(curline)
  cl=curline.split()
  M=int(cl[0])
  for ci in range(M):
    scalefac1=math.factorial(ci)/math.factorial(ci+1)
    for cj in range(M):
       curline=next(ff)
       cl=curline.split()
       idx=int(cl[0])
       modeval=float(cl[1])
       # rescale modeval
       new_modeval=modeval*math.factorial(cj)/math.factorial(cj+1)*scalefac1
       gg.write(str(idx)+' '+'{:e}'.format(new_modeval)+'\n')

  for x in ff.readlines():
    gg.write(x)

  ff.close()
  gg.close()

if __name__ == '__main__':
  # correct old shapelet modes to be used in new version of SAGECal 
  #args: old.modes new.modes
  import sys
  argc=len(sys.argv)
  if argc==3:
   read_shap(sys.argv[1],sys.argv[2])
  else:
   print("Usage %s old_model new_model"%sys.argv[0])
   print("new_model will be created")

  exit()
