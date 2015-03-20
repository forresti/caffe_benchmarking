
#qsub -I -q batch -l nodes=10 -l walltime=1:00:00 -A CSC103   
#aprun -n 10 -d 16 ./main 

#    7*7*64*3 = 9408 
#    11*11*96*3 = 34848
#    3*3*384*384 = 1327104 = 1.3M * 4 = 5.2MB
#    4096*4096 = 16777216 = 16.7M * 4 = 67MB
 

#TODO: loop over #procs

#for probSize in 9408 34848 1327104 16777216

#approx alexnet param sizes:
for probSize in 34848 614400 884736 1327104 16777216
do
    #aprun -n 10 -d 16 ./main $probSize 
    aprun -n 100 -d 16 ./main $probSize 
done


