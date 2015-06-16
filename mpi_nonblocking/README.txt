
#for Titan:
qsub -I -q batch -l nodes=10 -l walltime=1:00:00 -A CSC103   
aprun -n 10 -d 16 ./main


#for a19/a20:
mpirun --hostfile hostfiles/a19_a20_1slot.txt -np 2 ./main

#firebox:
/opt/openmpi/bin/mpirun --hostfile hostfiles/f1_f2_1slot.txt -np 2 ./main

message_size[num_messages] = {34848, 614400, 884736, 1327104, 884736};

RESULTS on Firebox-0 (15 nodes) so far:
- nonblocking: .24 GB/s
- blocking: .75 GB/s

Testing 16-bit on Firebox-0...
- blocking, change MPI_FLOAT to MPI_SHORT: 1.1 GB/s (w.r.t float) 
- blocking, change MPI_FLOAT to MPI_INT: .75 GB/s
- nonblocking, change MPI_FLOAT to MPI_SHORT: .24 GB/s 

#titan
Titan, 32 nodes:
- blocking: .52 GB/s (if I increase msg size, I get the same BW)
- nonblocking: (hangs)


Titan, 2 nodes:
- blocking: 2.5 GB/s

