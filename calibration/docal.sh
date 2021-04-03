#!/bin/bash
# copy new admm_rho
cp admm_rho_new.txt admm_rho.txt

# Note: data files are L_SB1.MS, L_SB2.MS ... L_SB8.MS

# select GPUs to use
export CUDA_VISIBLE_DEVICES=0,1
# run calibration
mpirun -np 3 /home/sarod/work/DIRAC/sagecal/build/dist/bin/sagecal-mpi -f 'L_SB[1-8].MS' -A 6 -P 3 -r 2.0 -s sky.txt -c cluster.txt -I DATA -O MODEL_DATA -p zsol -G admm_rho.txt -n 2 -t 10 -e 4 -g 2 -l 10 -m 7 -x 30 -F 1 -L 2 -V -N 0 -U 0

for MS in L_SB*.MS; do
 /home/sarod/work/excon/src/MS/excon -x 0 -m $MS -c MODEL_DATA -d 4800 
done

bash ./calmean.sh 'L_SB?.MS_I.fits' 1
python2 calmean_.py > X.out 2>&1
mv bar.fits orig/res.fits
