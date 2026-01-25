#!/bin/bash
# copy new admm_rho (created by calibenv.py)
cp admm_rho0.txt admm_rho_new.txt
cp admm_rho_new.txt admm_rho.txt

# Note: data files are L_SB1.MS, L_SB2.MS ... L_SB8.MS

# select GPUs to use
export CUDA_VISIBLE_DEVICES=0,1
# run calibration (with spatial regularization)
# -X lambda,mu,n0 (n0^2 basis),FISTA_iter,cadence
mpirun -np 3 /home/sarod/work/DIRAC/sagecal/build/dist/bin/sagecal-mpi_gpu -f 'L_SB[1-8].MS' -A 30 -P 3 -s sky.txt -c cluster.txt -I DATA -O MODEL_DATA -p zsol -G admm_rho.txt -n 2 -t 10 -e 4 -g 2 -l 10 -m 7 -x 30 -F 1 -L 2 -V -N 0 -U 0 -E 1  -X 0.1,1e-4,2,100,3

for MS in L_SB*.MS; do
 /home/sarod/work/excon/src/MS/excon -R 2 -x 0 -p 6 -m $MS -c MODEL_DATA -d 12000 
done

bash ./calmean.sh 'L_SB?.MS_I.fits' 1
python calmean_.py > X.out 2>&1
mv bar.fits orig/res.fits
