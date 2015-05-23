
GOALS
- [done] Understand MPI hostfiles
- [done] Allocate a local GPU ID to each global MPI rank

EXAMPLES

1 process per node:
mpirun --hostfile hostfiles/a19_a20_1slot.txt -np 2 ./main

2 processes per node:
mpirun --hostfile hostfiles/a19_a20_2slot.txt -np 4 ./main

