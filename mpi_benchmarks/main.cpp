#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
#include "mpi.h"
using namespace std;

//@return avg time for MPI_Allreduce
double benchmark_allreduce(int weight_count, int nRuns){

    //verified that the time spent in Malloc is trivial compared to Allreduce.
    float* weight_diff_local = (float*)malloc(weight_count * sizeof(float)); //sum of local weight diffs
    float* weight_diff = (float*)malloc(weight_count * sizeof(float)); //sum all weight diffs here

    //init to random noise
    for(int i=0; i<weight_count; i++){
        weight_diff_local[i] = (float)rand() / (float)RAND_MAX;
        weight_diff[i] = (float)rand() / (float)RAND_MAX;
    }

    double start = MPI_Wtime(); //in seconds

    for(int i=0; i<nRuns; i++){
        MPI_Allreduce(weight_diff_local, //send
                  weight_diff, //recv
                  weight_count, //count
                  MPI_FLOAT, 
                  MPI_SUM, //op
                  MPI_COMM_WORLD);
    }
    double end = MPI_Wtime();
    double elapsedTime = end - start; 
    elapsedTime = elapsedTime / nRuns;

    free(weight_diff);
    free(weight_diff_local);

    return elapsedTime;
}

//@param img_count = height*width*depth per image
double benchmark_scatter(int img_count, int img_per_proc, int nRuns){

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    float* img_fullbatch = NULL;
    float* img_localbatch =  (float*)malloc(img_count * img_per_proc * sizeof(float));

    if(rank == 0){
        img_fullbatch = (float*)malloc(img_count * img_per_proc * nproc * sizeof(float));

        //init to random noise
        for(int i=0; i < (img_count * img_per_proc * nproc); i++)
        //for(int i=0; i < (img_count * img_per_proc); i++)
        {
            img_fullbatch[i] = (float)rand() / (float)RAND_MAX;
        }
    }

    double start = MPI_Wtime(); //in seconds

    for(int i=0; i<nRuns; i++){
#if 1 
        //thx: http://stackoverflow.com/questions/13867809/how-are-mpi-scatter-and-mpi-gather-used-from-c
        MPI_Scatter(img_fullbatch, //send_data
                    img_count * img_per_proc, //send_count (per process)
                    //1, //tmp send_count
                    MPI_FLOAT, //send_datatype
                    img_localbatch, //recv_data 
                    img_count * img_per_proc, //recv_count (same as send_count?) 
                    //1, //tmp recv_count
                    MPI_FLOAT, //recv_datatype
                    0, //root
                    MPI_COMM_WORLD);
#endif
    }
    double end = MPI_Wtime();
    double elapsedTime = end - start; 
    elapsedTime = elapsedTime / nRuns;

    if(rank == 0)
        free(img_fullbatch);

    free(img_localbatch);

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

    //TODO: verify how MPI handles cmdline args (all procs seem to get the input args)
    //int weight_count = 3*3*384*384; // = 1327104 = 1.3M * 4 = 5.2MB
    int weight_count = 256*256*3; //for images
    if(argc == 2)
        weight_count = atoi(argv[1]);
    double GB_to_transfer = (double) weight_count * sizeof(float) / 1e9;

    if(rank == 0)
        printf("  weight_count = %d = %f GB\n", weight_count, GB_to_transfer);

    int nRuns = 50; //TODO: average time over nRuns.
    //double elapsedTime = benchmark_allreduce(weight_count, nRuns);

    int img_per_proc = 25;
    double elapsedTime = benchmark_scatter(weight_count, img_per_proc, nRuns); 

    //verified: all ranks get roughly the same elapsedTime.
    if(rank == 0)
        printf("    elapsedTime: %f sec, GB: %f, allreduce: %f GB/s \n", elapsedTime, GB_to_transfer, GB_to_transfer/elapsedTime);

    MPI_Finalize();

    return 0;
}
