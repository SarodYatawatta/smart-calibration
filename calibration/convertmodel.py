import math
import numpy as np
import lsmtool
from calibration_tools import *

# convert RA (degrees) to radians
def ra_to_rad(degval):
  return degval*math.pi/180.0
# convert Dec (degrees) to radians
def dec_to_rad(degval):
  return degval*math.pi/180.0
# convert arcsec to radians
def asec_to_rad(asec):
  return asec/(60*60)*math.pi/180.0

def read_skymodel(skymodel,sagecalsky,sagecalcluster,admm_rho='base.rho'):
  outsky=open(sagecalsky,'w+')
  outcluster=open(sagecalcluster,'w+')
  outrho=open(admm_rho,'w+')
  outsky.write('## LSM file\n')
  outsky.write("### Name  | RA (hr,min,sec) | DEC (deg,min,sec) | I | Q | U |  V | SI0 | SI1 | SI2 | RM | eX | eY | eP | freq0\n")
  outcluster.write('### Cluster file\n')
  s=lsmtool.load(skymodel)
  patches=s.getPatchNames()
  cluster_id=2 # cluster 1 is the target
  for patch in patches:
    s=lsmtool.load(skymodel)
    s.select('Patch == '+patch)
    t=s.table
    ra=ra_to_rad(np.array(t['Ra']))
    dec=dec_to_rad(np.array(t['Dec']))
    # use the following for generation targets close to A-team
    print('%s %f %f'%(patch,ra[0],dec[0]))
    stype=[x.encode('ascii') for x in np.array(t['Type'])]
    f0=np.array(t['ReferenceFrequency'])
    sI=np.array(t['I'])
    SpecI=np.array(t['SpectralIndex'][:,0])
    major=asec_to_rad(np.array(t['MajorAxis']))
    minor=asec_to_rad(np.array(t['MinorAxis']))
    pa=math.pi/2-(math.pi-dec_to_rad(np.array(t['Orientation'])))
    outcluster.write(str(cluster_id)+' 1')
    outrho.write(str(cluster_id)+' 1 1.0\n')
    for ci in range(ra.size):
        hh,mm,ss=radToRA(ra[ci])
        dd,dmm,dss=radToDec(dec[ci])
        if stype[ci].decode('ascii')=='GAUSSIAN':
          name='G'+patch+str(ci)
        else:
          name='P'+patch+str(ci)
        outsky.write(name+' '+str(hh)+' '+str(mm)+' '+str(ss)
          +' '+str(dd)+' '+str(dmm)+' '+str(dss)+' '+str(sI[ci])+' 0 0 0'
          +' '+str(SpecI[ci])+' 0 0 0'
          +' '+str(0.5*major[ci])+' '+str(0.5*minor[ci])+' '+str(pa[ci])
          +' '+str(f0[ci])+'\n')
        outcluster.write(' '+name)
    outcluster.write('\n')
    cluster_id+=1
  outsky.close()
  outcluster.close()
  outrho.close()
    


if __name__ == '__main__':
  # Convert sky model from DP3 to sagecal
  # Also creates cluster file (patch) and ADMM regularization rho
  import sys
  argc=len(sys.argv)
  if argc==4:
    read_skymodel(sys.argv[1],sys.argv[2],sys.argv[3])
  elif argc==5:
    read_skymodel(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
  else:
    print("Usage: %s skymodel sky.txt cluster.txt rho.txt"%sys.argv[0])
