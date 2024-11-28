#include <mpi.h>
#include <stdio.h>
#include <time.h>
               
#define PI25DT 3.141592653589793238462643

#define INTERVALS 100000000000

int main(int argc, char **argv)
{
    long int i, intervals = INTERVALS;
    double x, dx, f, sum, pi;
    double time2;
    int rank, size;
    MPI_Status status;
    MPI_Request req;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
    time_t time1 = clock();

    // Master process 0: receive partial results from slave processes.
    if (rank == 0) {
        sum = 0.0;
        dx = 1.0 / (double) INTERVALS;
        double temp;

        // Wait for slave processes to have finished.
        for (i = 1; i < size; i++) {
            MPI_Recv(&temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            if (status.MPI_ERROR != MPI_SUCCESS) {
                printf("An error occurred\n");
            }
            printf("I am the master, i have received %f from %d\n", temp, status.MPI_SOURCE);
            sum += temp;
            printf("I am the master, the total sum is %f\n", sum);
        }

        // Compute the final value.
        pi = dx*sum;

        time2 = (clock() - time1) / (double) CLOCKS_PER_SEC;

        printf("Computed PI %.24f\n", pi);
        printf("The true PI %.24f\n\n", PI25DT);
        printf("Elapsed time (s) = %.2lf\n", time2);
    }
    
    // Slave processes 1 to size-1: compute a sub-interval and send the result to the master.
    else {
        intervals = INTERVALS / (size - 1);
        sum = 0.0;
        dx = 1.0 / (double) INTERVALS;
        
        long start = intervals * (rank-1) + 1;
        long end = intervals * rank;

        printf("I am process %d, i am computing %ld intervals, from %ld to %ld\n", rank, end-start+1, start, end);
        for (i = start; i <= end; i++) {
            x = dx * ((double) (i - 0.5));
            f = 4.0 / (1.0 + x*x);
            //printf("f(%f)=%f\n", x, f);
            sum = sum + f;
            //printf("f from %f to %f = %f\n", dx * ((double) (start - 0.5)), x, sum);
        }

        printf("I am process %d, i have computed %f\n", rank, sum);

        MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}