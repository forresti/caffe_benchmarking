
#for Titan:
qsub -I -q batch -l nodes=10 -l walltime=1:00:00 -A CSC103   
aprun -n 10 -d 16 ./main


#for a19/a20:
mpirun --hostfile hostfiles/a19_a20_1slot.txt -np 2 ./main

#firebox:
/opt/openmpi/bin/mpirun --hostfile hostfiles/f1_f2_1slot.txt -np 2 ./main


RESULTS on Firebox (15 nodes) so far:
- nonblocking: .24 GB/s
- blocking: .75 GB/s

Testing 16-bit...
- blocking, change MPI_FLOAT to MPI_SHORT: 1.1 GB/s (w.r.t float) 
- blocking, change MPI_FLOAT to MPI_INT: .75 GB/s
- nonblocking, change MPI_FLOAT to MPI_SHORT: .24 GB/s 

