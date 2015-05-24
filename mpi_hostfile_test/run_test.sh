policy=--byslot #a19: {0, 1}; a20: {2, 3} ... this is default
#policy=--bynode #a19: {0, 2}; a20: {1, 3} ... round robin

#hostfile=hostfiles/a19_a20_2slot.txt
#hostfile=hostfiles/a18_a19_a20_allslot.txt
#hostfile=hostfiles/a18_8slot.txt
hostfile=hostfiles/f12_to_f16_1slot.txt

mpirun --hostfile $hostfile -np 5 $policy ./main

