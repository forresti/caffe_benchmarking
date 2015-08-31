
probSize=7000000 #28MB / 4 -> in bytes. NiN param file size
nRuns=1 #there's also an nRuns inside of the 'main' file... this is just for "meta-variance" analysis
mkdir results

#TODO: figure out if/how I want to measure "1 gpu per slot; 2 gpu per slot."

maxProcs=16
for ((nProcs=16; nProcs<=$maxProcs; nProcs=$nProcs*2))
do

    outF=results/out_${nProcs}_procs_allreduce.log
    rm $outF

    #hostfile=../mpi_hostfile_test/hostfiles/a18_8slot.txt
    #hostfile=../mpi_hostfile_test/hostfiles/f12_to_f16_1slot.txt
    hostfile=../mpi_hostfile_test/hostfiles/firebox_1slot.txt

    #firebox:
    for((i=0; i<$nRuns; i++))
    do
        /opt/openmpi/bin/mpirun --hostfile $hostfile -np $nProcs ./main $probSize >> $outF
    done
done


