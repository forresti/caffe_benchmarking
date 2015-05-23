#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
#include "mpi.h"
using namespace std;

struct eqstr
{
  bool operator()(const char* s1, const char* s2) const
  {
    return strcmp(s1, s2) == 0;
  }
};

//@param hostname = assume 1024 bytes, as in main()
//@param bufsz = size of hostname buffer
// thx for MPI_Gather example: mpi-forum.org/docs/mpi-1.1/mpi-11-html/node70.html
int select_gpu(int rank, int nproc, char* hostname, int bufsz){
    char* hostname_buf = (char*)malloc( nproc * bufsz * sizeof(char)); 
    int gpu_id[nproc];

    map<string, int> host_counts; //TODO: just use normal 'map'? (don't need c++11 then)

    MPI_Gather( hostname, //send
                bufsz,
                MPI_CHAR,
                hostname_buf, //recv
                bufsz,
                MPI_CHAR,
                0, //gather to rank 0
                MPI_COMM_WORLD);
      

    if(rank == 0){
        //THE BRAINS OF THE OPERATION ... assign GPU ID to each rank
        for(int p=0; p<nproc; p++){
            string h( &hostname_buf[p*bufsz] ); //h[0:bufsz] = hostname_buf[p][0:bufsz]
            if( host_counts.find(h) == host_counts.end() ){
                //haven't seen this hostname before
                gpu_id[p] = 0;
                host_counts[h] = 1;
            }
            else{ 
                gpu_id[p] = host_counts[h];
                host_counts[h]++;
            }
            //printf("gpu_id[%d] = %d, h=%s \n", p, gpu_id[p], h.c_str());
        }
    }

    int my_gpu_id;
    MPI_Scatter(gpu_id, //sendbuf
                1, //num to send
                MPI_INT,
                &my_gpu_id, //recvbuf
                1, //num to recv
                MPI_INT,
                0, //root
                MPI_COMM_WORLD);
    free(hostname_buf);

    return my_gpu_id;
}


//@param argv = [(optional) size of data to transfer]
int main (int argc, char **argv)
{
    int rank, nproc;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    //thx: stackoverflow.com/questions/504810
    int bufsz = 1024;
    char hostname[bufsz];
    hostname[bufsz-1] = '\0';
    gethostname(hostname, bufsz);

    int gpu_id = select_gpu(rank, nproc, hostname, bufsz);

    printf(" rank = %d, hostname = %s, gpu_id = %d \n", rank, hostname, gpu_id);


    MPI_Finalize();

    return 0;
}
