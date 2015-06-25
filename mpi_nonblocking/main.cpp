#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
#include <assert.h>
#include "mpi.h"
using namespace std;

enum communication_t {BLOCKING=1, NONBLOCKING=2, POINT2POINT=3}; 

int comm = BLOCKING;
//int comm = POINT2POINT;

//@return avg time for MPI_Allreduce
double benchmark_allreduce(int nRuns, int* out_weight_count){
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // ** PROBLEM DIMS **
    //const int num_messages = 1;
    //int message_size[num_messages] = {1327104};

    //const int num_messages = 5;
    //int message_size[num_messages] = {34848, 614400, 884736, 1327104, 884736}; //sorta like alexnet conv weights
    //int message_size[num_messages] = {4194304, 4194304, 4194304, 4194304, 4194304}; //4mb each

    const int num_messages = 14;
    float message_size_fl[num_messages] = {0.088, 0.02343, 0.0351, 0.0234, 0.585, 0.0469, 0.0625, 0.0469, 0.281, 0.125, 0.125, 0.563, 0.281, 1.46}; //for my 4MB NiN tripleCCCP ... in MB
    int message_size[num_messages];
    for(int m=0; m<num_messages; m++){
        message_size[m] = (int) 1024*1024*message_size_fl[m];
    }
  // ** INIT DATA **
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

  // ** PERFORM BENCHMARK **
    double start = MPI_Wtime(); //in seconds

    for(int i=0; i<nRuns; i++){
        if(comm == BLOCKING){
            for(int m=0; m<num_messages; m++){
                MPI_Allreduce(MPI_IN_PLACE, //weight_diff_local, //send
                          weight_diff[m], //recv
                          message_size[m], //count
                          MPI_FLOAT,
                          MPI_SUM, //op
                          MPI_COMM_WORLD);
            }
        }
        else if(comm == NONBLOCKING){
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
        }
        else if(comm == POINT2POINT){
            //send data from each worker to root 
            //intentionally NOT doing gather/scatter, becuase we're emulating "async param setver"
            if(rank != 0){
                assert(num_messages == 1); //haven't set this up for multiple messages yet
                int m = 0;
                MPI_Send(weight_diff[m], 
                         message_size[m],
                         MPI_FLOAT,
                         0, //dest = root
                         0, //tag
                         MPI_COMM_WORLD);
            }
            else //rank == 0
            {
                for(int i=1; i<nproc; i++){
                   int m = 0;
                    MPI_Recv(weight_diff[m],
                             message_size[m],
                             MPI_FLOAT,
                             MPI_ANY_SOURCE, //source
                             MPI_ANY_TAG, //tag
                             MPI_COMM_WORLD,
                             &statuses[m]); //not really saving status...
                }
            }
        }
        else{
            printf("unknown communication type. aborting.\n");
            exit(0);
        }
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
