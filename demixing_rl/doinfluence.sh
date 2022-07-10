#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

let "ci = 0"; 
while [ $ci -le 0 ]; do # -le 8 for all freqs
 if [ $ci -le 9 ]; then
  MS="L_SB"$ci".MS";
 elif [ $ci -le 99 ]; then
  MS="L_SB"$ci".MS";
 fi
 python ./readcorr.py $MS smalluvw.txt;
 python  ./analysis.py ./sky.txt ./cluster_epi.txt ./smalluvw.txt ./admm_rho_epi.txt $MS.solutions 0.1 4 # last parameters: alpha number_of_parallel_jobs
 python ./writecorr.py $MS fff;
 # -x 2 for fullpol
 /home/sarod/work/excon/src/MS/excon -x 0 -m $MS -c CORRECTED_DATA -d 128 -p 20
 let "ci = $ci + 1";
done

#bash ./calmean.sh 'L_SB?.MS_I.fits' 1
#python2 calmean_.py > X.out 2>&1
#mv bar.fits ./influenceI.fits
cp L_SB0.MS_I.fits ./influenceI.fits

#bash ./calmean.sh 'L_SB?.MS_Q.fits' 1
#python2 calmean_.py
#mv bar.fits ./influenceQ.fits

#bash ./calmean.sh 'L_SB?.MS_U.fits' 1
#python2 calmean_.py
#mv bar.fits ./influenceU.fits

#bash ./calmean.sh 'L_SB?.MS_V.fits' 1
#python2 calmean_.py
#mv bar.fits ./influenceV.fits
