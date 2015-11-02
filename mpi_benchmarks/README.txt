
ln -s Makefile_titan Makefile

#for Titan:
qsub -I -q batch -l nodes=10 -l walltime=1:00:00 -A CSC103   
aprun -n 10 -d 16 ./main


#for a19/a20:
mpirun --hostfile hostfiles/a19_a20_1slot.txt -np 2 ./main

#firebox:
/opt/openmpi/bin/mpirun --hostfile hostfiles/f1_f2_1slot.txt -np 2 ./main


