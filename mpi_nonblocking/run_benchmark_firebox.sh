
#a18 / a19 / a20
#mpirun --hostfile hostfiles/a18_8slot.txt -np 8 ./main 

#hostfile=../mpi_hostfile_test/hostfiles/f12_to_f16_1slot.txt
hostfile=../mpi_hostfile_test/hostfiles/firebox_1slot.txt

#firebox:
/opt/openmpi/bin/mpirun --hostfile $hostfile -np 15 ./main 



