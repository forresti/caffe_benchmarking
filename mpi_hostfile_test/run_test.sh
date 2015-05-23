policy=--byslot #a19: {0, 1}; a20: {2, 3} ... this is default
#policy=--bynode #a19: {0, 2}; a20: {1, 3} ... round robin

mpirun --hostfile hostfiles/a19_a20_2slot.txt -np 4 $policy ./main

