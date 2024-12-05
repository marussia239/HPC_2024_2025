#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <omp.h>
#include <mpi.h>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 5000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1500 // Maximum number of iterations

using namespace std;

int main(int argc, char **argv)
{
    int *const image = new int[HEIGHT * WIDTH];
    long int i, pos, n_intervals;
    int rank, size;
    MPI_Status status;
    MPI_Request req;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    n_intervals = HEIGHT * WIDTH / (size - 1);

    if (rank == 0) {
        int *temp = new int[n_intervals];

        const auto start = chrono::steady_clock::now();

        // Wait for slave processes to have finished.
        for (i = 1; i < size; i++) {

            MPI_Recv(temp, n_intervals, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            long start = n_intervals * (status.MPI_SOURCE-1) + 1;
            long end = n_intervals * status.MPI_SOURCE;
            if (status.MPI_ERROR != MPI_SUCCESS) {
                cout << "An error occurred" << endl;
            }

            // Store the data received inside the main image
            for (long pos = start; pos <= end; pos++) {
                image[pos] = temp[pos - start];
            }
        }

        const auto end = chrono::steady_clock::now();
        cout << "Time elapsed: "
            << chrono::duration_cast<chrono::milliseconds>(end - start).count()
            << " seconds." << endl;

        // Write the result to a file
        ofstream matrix_out;

        if (argc < 2)
        {
            cout << "Please specify the output file as a parameter." << endl;
            return -1;
        }

        matrix_out.open(argv[1], ios::trunc);
        if (!matrix_out.is_open())
        {
            cout << "Unable to open file." << endl;
            return -2;
        }

        const auto print_start = chrono::steady_clock::now();
        for (int row = 0; row < HEIGHT; row++)
        {
            for (int col = 0; col < WIDTH; col++)
            {
                matrix_out << image[row * WIDTH + col];

                if (col < WIDTH - 1)
                    matrix_out << ',';
            }
            if (row < HEIGHT - 1)
                matrix_out << endl;
        }
        matrix_out.close();

        const auto print_end = chrono::steady_clock::now();
        cout << "Time elapsed: "
            << chrono::duration_cast<chrono::milliseconds>(print_end - print_start).count()
            << " seconds." << endl;
    }

    else {
        n_intervals = HEIGHT * WIDTH / (size - 1);
        
        long start = n_intervals * (rank-1) + 1;
        long end = n_intervals * rank;
        int *temp = new int[n_intervals];

        #pragma omp parallel for
        for (pos = 0; pos < n_intervals; pos++) {
            temp[pos] = 0;

            const int row = (start + pos) / WIDTH;
            const int col = (start + pos) % WIDTH;
            const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

            // z = z^2 + c
            complex<double> z(0, 0);

            for (int i = 1; i <= ITERATIONS; i++)
            {
                z = pow(z, 2) + c;

                // If it is convergent
                if (abs(z) >= 2)
                {
                    temp[pos] = i;
                    break;
                }
            }
        }

        MPI_Send(temp, n_intervals, MPI_INT, 0, 0, MPI_COMM_WORLD);

        delete[] temp;
    }

    delete[] image; // It's here for coding style, but useless
    MPI_Finalize();
    return 0;
}