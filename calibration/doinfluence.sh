#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

#let "ci = 1";
let "ci = 4"; # only use mid frequency
while [ $ci -le 4 ]; do # -le 8 for all freqs
 if [ $ci -le 9 ]; then
  MS="L_SB"$ci".MS";
 elif [ $ci -le 99 ]; then
  MS="L_SB"$ci".MS";
 fi
 python2 ./readcorr.py $MS smalluvw.txt;
 #matlab -nodesktop -nosplash -r "analysis_uvwdir_loop2 ./sky.txt ./cluster.txt ./smalluvw.txt ./admm_rho.txt $MS.solutions"
 python  ./analysis.py ./sky.txt ./cluster.txt ./smalluvw.txt ./admm_rho.txt $MS.solutions
 python2 ./writecorr.py $MS fff;
 # -x 2 for fullpol
 /home/sarod/work/excon/src/MS/excon -x 0 -m $MS -c CORRECTED_DATA -d 128 -p 20
 let "ci = $ci + 1";
done

#bash ./calmean.sh 'L_SB?.MS_I.fits' 1
#python2 calmean_.py > X.out 2>&1
#mv bar.fits orig/influenceI.fits
cp L_SB4.MS_I.fits orig/influenceI.fits

#bash ./calmean.sh 'L_SB?.MS_Q.fits' 1
#python2 calmean_.py
#mv bar.fits orig/influenceQ.fits

#bash ./calmean.sh 'L_SB?.MS_U.fits' 1
#python2 calmean_.py
#mv bar.fits orig/influenceU.fits

#bash ./calmean.sh 'L_SB?.MS_V.fits' 1
#python2 calmean_.py
#mv bar.fits orig/influenceV.fits
