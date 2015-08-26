
#qsub -I -q debug -l nodes=128 -l walltime=1:00:00 -A CSC103   
#aprun -n 8 -d 16 ./main 

probSize=7000000 #28MB / 4 -> in bytes. NiN param file size
nRuns=10 #there's also an nRuns inside of the 'main' file... this is just for "meta-variance" analysis

for ((nProcs=2; nProcs<=128; nProcs=$nProcs*2))
do

    outF=out_${nprocs}_procs_allreduce.log
    rm $outF
    for((i=0; i<$nRuns; i++))
    do
        echo $nProcs
        #aprun -n $nProcs -d 16 ./main $probSize >> $outF
    done
done

