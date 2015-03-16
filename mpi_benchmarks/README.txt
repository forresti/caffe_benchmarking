
qsub -I -q batch -l nodes=10 -l walltime=1:00:00 -A CSC103   
aprun -n 10 -d 16 ./main

