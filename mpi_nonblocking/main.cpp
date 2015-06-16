#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
#include "mpi.h"
using namespace std;

//#define NONBLOCKING

//@return avg time for MPI_Allreduce
double benchmark_allreduce(int nRuns, int* out_weight_count){

    //const int num_messages = 1;
    //int message_size[num_messages] = {1327104};

    const int num_messages = 5;
    int message_size[num_messages] = {34848, 614400, 884736, 1327104, 884736}; //sorta like alexnet conv weights
    //int message_size[num_messages] = {4194304, 4194304, 4194304, 4194304, 4194304}; //4mb each

    float** weight_diff = (float**)malloc(num_messages * sizeof(float*));
    int weight_count_ = 0;

    //verified that the time spent in Malloc is trivial compared to Allreduce.
    for(int m=0; m<num_messages; m++){
        weight_diff[m] = (float*)malloc(message_size[m] * sizeof(float)); //sum all weight diffs here
        weight_count_ += message_size[m];

        //init to random noise
        for(int i=0; i<message_size[m]; i++){
            weight_diff[m][i] = (float)rand() / (float)RAND_MAX;
        }
    }
    out_weight_count[0] = weight_count_;

    MPI_Request requests[num_messages];
    MPI_Status statuses[num_messages];

    double start = MPI_Wtime(); //in seconds

    for(int i=0; i<nRuns; i++){
#ifndef NONBLOCKING
        for(int m=0; m<num_messages; m++){
            MPI_Allreduce(MPI_IN_PLACE, //weight_diff_local, //send
                      weight_diff[m], //recv
                      message_size[m], //count
                      MPI_FLOAT,
                      MPI_SUM, //op
                      MPI_COMM_WORLD);
        }
#else
        for(int m=0; m<num_messages; m++){
            MPI_Iallreduce(MPI_IN_PLACE, //weight_diff_local, //send
                      weight_diff[m], //recv
                      message_size[m], //count
                      MPI_FLOAT,
                      MPI_SUM, //op
                      MPI_COMM_WORLD,
                      &requests[m]);

            //int flag;
            //MPI_Test(&requests[m], &flag, &statuses[m]); //Mark Hoemmen says this may be necessary to start the message (this doesn't seem to increase BW)
        }
        //MPI_Waitall(num_messages, requests, statuses);
        for(int m=0; m<num_messages; m++){
            MPI_Wait(&requests[m], &statuses[m]);
        }
#endif
    }

    double end = MPI_Wtime();
    double elapsedTime = end - start; 
    elapsedTime = elapsedTime / nRuns;

    for(int m=0; m<num_messages; m++){
        free(weight_diff[m]);
    }
    free(weight_diff);

    return elapsedTime;
}

//@param argv = [(optional) size of data to transfer]
int main (int argc, char **argv)
{
    int rank, nproc;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int weight_count;

    int nRuns = 50;
    double elapsedTime = benchmark_allreduce(nRuns, &weight_count);
    double GB_to_transfer = (double) weight_count * sizeof(float) / 1e9;

    //verified: all ranks get roughly the same elapsedTime.
    if(rank == 0)
        printf("    elapsedTime: %f sec, GB: %f, allreduce: %f GB/s \n", elapsedTime, GB_to_transfer, GB_to_transfer/elapsedTime);

    MPI_Finalize();

    return 0;
}
