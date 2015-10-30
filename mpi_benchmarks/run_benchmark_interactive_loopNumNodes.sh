
#qsub -I -q debug -l nodes=128 -l walltime=1:00:00 -A CSC103   
#aprun -n 8 -d 16 ./main 

#probSize=7000000 #28MB / 4 -> in bytes. NiN param file size
#probSize=8750 #cccp1: 0.035MB / 4 -> in bytes
probSize=140625 #cccp5: 0.5625MB / 4 -> in bytes
outDir=results_${probSize}_bytes
nRuns=2 #there's also an nRuns inside of the 'main' file... this is just for "meta-variance" analysis
mkdir $outDir

for ((nProcs=2; nProcs<=128; nProcs=$nProcs*2))
do

    outF=out_${nProcs}_procs.log
    rm $outF
    for((i=0; i<$nRuns; i++))
    do
        echo $nProcs
        aprun -n $nProcs -d 16 ./main $probSize >> $outDir/$outF
    done
done

