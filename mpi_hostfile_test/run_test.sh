#policy=--byslot #a19: {0, 1}; a20: {2, 3} ... this is default
policy=--bynode #a19: {0, 2}; a20: {1, 3} ... round robin

#hostfile=hostfiles/a19_a20_2slot.txt
hostfile=hostfiles/a18_a19_a20_allslot.txt

mpirun --hostfile $hostfile -np 12 $policy ./main

