#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
#include "mpi.h"
using namespace std;

//@param argv = [(optional) size of data to transfer]
int main (int argc, char **argv)
{
    int rank, nproc;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    //thx: stackoverflow.com/questions/504810
    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);

    printf(" rank = %d, hostname = %s \n", rank, hostname);

    MPI_Finalize();

    return 0;
}
