#!/bin/bash

# simulate solutions, sky model
#python simulate.py

export CUDA_VISIBLE_DEVICES=0,1

declare -a freqlist=(000.000 115   125   135   145   155   165   175   185);
let "ci = 1";
while [ $ci -le 8 ]; do
 if [ $ci -le 9 ]; then
  MS="L_SB"$ci".MS";
 elif [ $ci -le 99 ]; then
  MS="L_SB"$ci".MS";
 fi
 rsync -av ~/small.ms/ $MS;
 python2 ./changefreq.py $MS ${freqlist[$ci]}"e6"
 /home/sarod/work/DIRAC/sagecal/build/dist/bin/sagecal  -d $MS -s sky0.txt -c cluster0.txt -p $MS'.S.solutions' -t 10 -O DATA -a 1 -B 1

 python2 ./addnoise.py $MS
 /home/sarod/work/excon/src/MS/excon -x 0 -m $MS -c DATA -d 4800

 let "ci = $ci + 1";
done


bash ./calmean.sh 'L_SB?.MS_I.fits' 1
python2 calmean_.py > X.out 2>&1
mv bar.fits orig/data.fits
# bash ~/work/python/eor/calmean.sh 'L_SB?.MS_I.fits' 1
