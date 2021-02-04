# usage bash ~/calmean.sh 'node0*' 0.1
GL="calmean_.py"
echo "from pyrap.images import image" > $GL;
echo "import os,math" >> $GL;
echo "import pyfits" >> $GL;
let "ct = 1";

## box to calculate variance
XLOW=1;
XHIGH=10;
YLOW=1;
YHIGH=10;

### variance cutoff
if [ -z "$2" ]; then
 vmax=0.01;
else
 vmax=$2;
fi
### catch images with zero data
vmin=0.000000000001;

echo "wgt=0;" >> $GL;
echo "bmaj=0;" >> $GL;
echo "bmin=0;" >> $GL;
echo "bpax=0;" >> $GL;
echo "bpay=0;" >> $GL;
echo "freq0=0;" >> $GL;

echo "ii=0;" >> $GL;
if [ -z "$1" ]; then
 flist=`find L*fits`;
else
 flist=`find $1`;
fi
for f in $flist; do
  if [ $ct == 1 ]; then
    echo "os.spawnvp(os.P_WAIT,'cp',['cp','"$f"','foo.fits'])" >> $GL;
  fi
  echo "itmp=image(\""$f"\").getdata()" >> $GL;
  # calcualte mean 
  echo "mt=itmp[0,0,"$XLOW":"$XHIGH","$YLOW":"$YHIGH"].mean()" >> $GL;
  # no mean 
  echo "mt=0.0" >> $GL;
  # make zero mean data
  echo "itmp=itmp-mt" >> $GL;
  # calcualte variance
  #echo "wt=itmp[0,0,"$XLOW":"$XHIGH","$YLOW":"$YHIGH"].std()" >> $GL;
  # no variance weight
  echo "wt=0.99999" >> $GL;
  #echo "print \""$f"\"" >> $GL;
  echo "if (not math.isnan(wt)) and wt < "$vmax" and wt > "$vmin":" >> $GL;
  echo "  sigma=1.0/(wt*wt)" >> $GL;
  echo "  wgt=wgt+sigma" >> $GL;
  echo "  ii = ii + itmp*sigma" >> $GL;
  # now use pyfits to open same file
  echo "  itmp=pyfits.open(\""$f"\")" >> $GL;
  echo "  try:" >> $GL;
  echo "    mybmaj=itmp[0].header['BMAJ']" >> $GL;
  echo "    mybmin=itmp[0].header['BMIN']" >> $GL;
  echo "    mybpa=itmp[0].header['BPA']" >> $GL;
  #echo "    myfreq=itmp[0].header['RESTFREQ']" >> $GL;
  #echo "    myfreq=itmp[0].header['RESTFRQ']" >> $GL;
  echo "    myfreq=itmp[0].header['CRVAL3']" >> $GL;
  echo "    bmaj=bmaj+mybmaj*sigma" >> $GL;
  echo "    bmin=bmin+mybmin*sigma" >> $GL;
  echo "    bpax=bpax+math.cos(mybpa*math.pi/180.0)*sigma" >> $GL;
  echo "    bpay=bpay+math.sin(mybpa*math.pi/180.0)*sigma" >> $GL;
  echo "    freq0=freq0+myfreq*sigma" >> $GL;
  echo "  except:" >> $GL;
  echo "    pass" >> $GL;
  echo "else:" >> $GL;
  echo "  print ( \"reject '"$f" '\"+str(wt)" \) >> $GL;
  
  let "ct = $ct + 1";
done

echo "if wgt==0.0 :" >> $GL;
echo " wgt=1.0" >> $GL;
echo "ii=ii/wgt" >> $GL;
echo "jj=image('foo.fits',overwrite=True)" >> $GL;
echo "jj.saveas('foo.img')" >> $GL;
echo "jj=image('foo.img')" >> $GL;
echo "jj.putdata(ii)" >> $GL;
echo "jj.tofits('bar.fits',velocity=False,bitpix=-32)" >> $GL;
echo "itmp=pyfits.open('bar.fits','update')" >> $GL;
echo "data=pyfits.getdata('bar.fits')" >> $GL;
echo "prihdr=itmp[0].header" >> $GL;
#echo "prihdr.update('BMAJ',bmaj/wgt)" >> $GL;
echo "prihdr['BMAJ']=bmaj/wgt" >> $GL;
#echo "prihdr.update('BMIN',bmin/wgt)" >> $GL;
echo "prihdr['BMIN']=bmin/wgt" >> $GL;
#echo "prihdr.update('BPA',math.atan2(bpay/wgt,bpax/wgt)*180.0/math.pi)" >> $GL;
echo "prihdr['BPA']=math.atan2(bpay/wgt,bpax/wgt)*180.0/math.pi" >> $GL;
#echo "prihdr.update('RESTFREQ',freq0/wgt)" >> $GL;
echo "prihdr['RESTFREQ']=freq0/wgt" >> $GL;
# also update FREQ axis
echo "prihdr['CRVAL3']=freq0/wgt" >> $GL;
echo "pyfits.update('bar.fits',data,prihdr)" >> $GL;
