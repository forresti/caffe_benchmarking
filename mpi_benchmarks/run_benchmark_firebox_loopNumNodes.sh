
probSize=7000000 #28MB / 4 -> in bytes. NiN param file size
nRuns=1 #there's also an nRuns inside of the 'allreduce_benchmark' file... this is just for "meta-variance" analysis
outDir=results_allreduce
#outDir=results_scatter_eth

mkdir $outDir

#TODO: figure out if/how I want to measure "1 gpu per slot; 2 gpu per slot."

#maxProcs=8
#for ((nProcs=2; nProcs<=$maxProcs; nProcs=$nProcs*2))
for nProcs in 2 4 8 12
do

    outF=$outDir/out_${nProcs}_procs.log
    rm $outF

    #hostfile=hostfiles/a18_8slot.txt
    #hostfile=hostfiles/f12_to_f16_1slot.txt
    hostfile=hostfiles/firebox_1slot.txt
    #hostfile=hostfiles/firebox_1slot_eth.txt #1GB/s eth0 (MPI stuff hangs when using this)

    echo "nProcs: " $nProcs

    #firebox:
    for((i=0; i<$nRuns; i++))
    do
        #defaults to Infiniband:
        #/opt/openmpi/bin/mpirun --hostfile $hostfile -np $nProcs ./allreduce_benchmark $probSize >> $outF

        #56 GB/s infiniband
        #/opt/openmpi/bin/mpirun --mca btl_tcp_if_include ib0 --mca btl tcp --hostfile $hostfile -np $nProcs ./allreduce_benchmark $probSize >> $outF 

        #40GB/s ethernet (eth2):
        #/opt/openmpi/bin/mpirun --mca btl_tcp_if_include eth2 --mca btl tcp --hostfile $hostfile -np $nProcs ./allreduce_benchmark $probSize >> $outF


        #Kostadin's version 12/14/15 (can also use `pml cm`, but it is slow or hangs for messages > 1MB)
        /opt/openmpi/bin/mpirun --mca btl_openib_if_include mlx4_1:1 --mca btl self,openib --mca pml ob1 --hostfile $hostfile -np $nProcs ./allreduce_benchmark $probSize #>> $outF

    done
done


