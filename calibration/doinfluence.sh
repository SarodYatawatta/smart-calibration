#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1


mpirun -np 3 /home/sarod/work/DIRAC/sagecal/build/dist/bin/sagecal-mpi_gpu -f 'L_SB[1-8].MS' -A 30 -P 3 -s sky.txt -c cluster.txt -I DATA -O CORRECTED_DATA -p zsol -G admm_rho.txt -n 2 -t 10 -e 4 -g 2 -l 10 -m 7 -x 30 -F 1 -L 2 -V -N 0 -U 0 -E 1 -i 1

for MS in L_SB*.MS; do
 /home/sarod/work/excon/src/MS/excon -R 2 -x 0 -p 6 -m $MS -c CORRECTED_DATA -d 12000 
done

bash ./calmean.sh 'L_SB?.MS_I.fits' 1
python calmean_.py > X.out 2>&1
mv bar.fits orig/influenceI.fits
