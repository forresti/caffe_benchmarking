
#for Titan:
qsub -I -q batch -l nodes=10 -l walltime=1:00:00 -A CSC103   
aprun -n 10 -d 16 ./main


#for a19/a20:
mpirun --hostfile hostfiles/a19_a20_1slot.txt -np 2 ./main

#firebox:
/opt/openmpi/bin/mpirun --hostfile hostfiles/f1_f2_1slot.txt -np 2 ./main

message_size[num_messages] = {34848, 614400, 884736, 1327104, 884736};

RESULTS on Firebox-0 (15 nodes) Allreduce so far:
- nonblocking: .24 GB/s
- blocking: .75 GB/s

Testing 16-bit on Firebox-0...
- blocking, change MPI_FLOAT to MPI_SHORT: 1.1 GB/s (w.r.t float) 
- blocking, change MPI_FLOAT to MPI_INT: .75 GB/s
- nonblocking, change MPI_FLOAT to MPI_SHORT: .24 GB/s 

#titan
Titan Allreduce:
- [2 nodes] blocking: 2.5 GB/s
- [32 nodes] blocking: .52 GB/s (if I increase msg size, I get the same BW)
- [64 nodes] blocking: .51 GB/s 
- [64 nodes] blocking: 1.3 GB/s (16bit w.r.t. float)
- [128 nodes] blocking: .49 GB/s
- [256 nodes] blocking: .47 GB/s
    elapsedTime: 0.177817 sec, GB: 0.083886, allreduce: 0.471755 GB/s
- nonblocking: (hangs)

  w/ 14 tiny messages (my 4MB NiN)
- [16 nodes] .69 GB/s
- [32 nodes] .58 GB/s
   -> no need to batch the messages, at least for now.

Titan point-to-point:
  (only measuring workers->root)
- [2 nodes] 5.3 GB/s
- [8 nodes] .72 GB/s
- [16 nodes] .21 GB/s
- [32 nodes] .085 GB/s
