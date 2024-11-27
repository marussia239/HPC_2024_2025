#include <mpi.h>
#include <stdio.h>

#define ndata 4

int main( int argc, char *argv[] )
{
    int rank, size, len;
    MPI_Status status;
    MPI_Request req;

    float a[ndata];
    float b[ndata];
    float c = 0;
    
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    for(int i=0;i<ndata;++i){
        a[i] = rank;
    }

    MPI_Sendrecv(a, ndata, MPI_REAL, (rank+1)%ndata, 0, b, ndata, MPI_REAL, 
    (rank-1+ndata)%ndata, 0, MPI_COMM_WORLD, status);
    c = c + b[0];
    printf(" I am task %d and I the sum is %1.2f \n", rank, b[0]);

    MPI_Finalize();
    return 0;
}
