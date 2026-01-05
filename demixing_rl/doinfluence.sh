#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

mpirun -np 3 /home/sarod/work/DIRAC/sagecal/build/dist/bin/sagecal-mpi_gpu -f 'L_SB*.MS' -A 30 -P 2 -s sky.txt -c cluster_epi.txt -I DATA -O CORRECTED_DATA -p zsol -G admm_rho_epi.txt -n 2 -t 10 -e 4 -g 2 -l 10 -m 7 -x 30 -F 1 -L 2 -V -N 0 -U 0 -E 1 -i 1
let "ci = 0"; 
while [ $ci -le 2 ]; do # -le 2 for all freqs
 if [ $ci -le 9 ]; then
  MS="L_SB"$ci".MS";
 elif [ $ci -le 99 ]; then
  MS="L_SB"$ci".MS";
 fi
 # -x 2 for fullpol
 /home/sarod/work/excon/src/MS/excon -t 4 -x 0 -m $MS -c CORRECTED_DATA -d 128 -p 20
 let "ci = $ci + 1";
done

bash ./calmean.sh 'L_SB?.MS_I.fits' 1
python calmean_.py > X.out 2>&1
mv bar.fits ./influenceI.fits

#bash ./calmean.sh 'L_SB?.MS_Q.fits' 1
#python2 calmean_.py
#mv bar.fits ./influenceQ.fits

#bash ./calmean.sh 'L_SB?.MS_U.fits' 1
#python2 calmean_.py
#mv bar.fits ./influenceU.fits

#bash ./calmean.sh 'L_SB?.MS_V.fits' 1
#python2 calmean_.py
#mv bar.fits ./influenceV.fits
