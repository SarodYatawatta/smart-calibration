#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

# mandetory arguments for this script: freq_low(MHz) freq_high(MHz) ra0 dec0 time_slots

let "ci = 0"; 
while [ $ci -le 0 ]; do # -le 8 for all freqs
 if [ $ci -le 9 ]; then
  MS="L_SB"$ci".MS";
 elif [ $ci -le 99 ]; then
  MS="L_SB"$ci".MS";
 fi
 python  ./analysis_torch.py ./sky.txt ./cluster_epi.txt $MS ./admm_rho_epi.txt $MS.solutions ./zsol $1 $2 $3 $4 $5 1 # last parameter: number_of_parallel_jobs(=1 for GPU version)
 # -x 2 for fullpol
 /home/sarod/work/excon/src/MS/excon -t 4 -x 0 -m $MS -c CORRECTED_DATA -d 128 -p 20
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
