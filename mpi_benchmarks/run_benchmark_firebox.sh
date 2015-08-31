
#qsub -I -q batch -l nodes=10 -l walltime=1:00:00 -A CSC103   
#aprun -n 10 -d 16 ./main 

#    7*7*64*3 = 9408 
#    11*11*96*3 = 34848
#    3*3*384*384 = 1327104 = 1.3M * 4 = 5.2MB
#    4096*4096 = 16777216 = 16.7M * 4 = 67MB
 

#TODO: loop over #procs

#approx alexnet param sizes:
for probSize in 34848 614400 884736 1327104 16777216
#for probSize in 34848
do
    #a18 / a19 / a20
    #mpirun --hostfile hostfiles/a18_8slot.txt -np 8 ./main $probSize

    #hostfile=../mpi_hostfile_test/hostfiles/a18_8slot.txt
    #hostfile=../mpi_hostfile_test/hostfiles/f12_to_f16_1slot.txt
    hostfile=../mpi_hostfile_test/hostfiles/firebox_1slot.txt
    #firebox:
    /opt/openmpi/bin/mpirun --hostfile $hostfile -np 8 ./main $probSize
done


