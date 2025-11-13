/*
 * assign2_1.cc
 *
 * Contains code for setting up and finishing the simulation.
 *
 * NOTE: YOU SHOULD NOT CHANGE THIS FILE
 *
 */

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <limits>

#include "file.hh"
#include "timer.hh"
#include "simulate.hh"

using namespace std;


/* Initializes the given double array with a sinus function. */
void fill(double *array, int offset, int range, double sample_start,
          double sample_end) {
    int i;
    float dx;

    dx = (sample_end - sample_start) / range;
    for (i = 0; i < range; i++) {
        array[i + offset] = sin(sample_start + i * dx);
    }
}


/* Entry point for the program. */
int main(int argc, char* argv[]) {
    // Check if the correct arguments are given
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " i_max t_max block_size" << endl;
        cout << " - i_max: number of discrete amplitude points, should be >2"
             << endl;
        cout << " - t_max: number of discrete timesteps, should be >=1"
             << endl;
        cout << " - block_size: number of CUDA-threads in each cuda-block. "
                "Should be a multiple of 32" << endl;
        return EXIT_FAILURE;
    }

    // Parse the input
    const long i_max      = strtol(argv[1], NULL, 10);
    const long t_max      = strtol(argv[2], NULL, 10);
    const long block_size = strtol(argv[3], NULL, 10);
    if (i_max < 3) {
        cerr << "Argument error: i_max should be and integer of >2." << endl;
        return EXIT_FAILURE;
    }
    if (t_max < 1) {
        cerr << "Argument error: t_max should be and integer of >=1." << endl;
        return EXIT_FAILURE;
    }
    if (block_size < 1) {
        cerr << "Argument error: block_size should be and integer of >=1."
             << endl;
        return EXIT_FAILURE;
    }

    // Initialize the timer
    timer waveTimer("wave timer");

    // Declare the arrays
    double *old_array     = new double[i_max]();
    double *current_array = new double[i_max]();
    double *next_array    = new double[i_max]();

    // Fill the first two with a sinus
    fill(old_array, 1, i_max/4, 0, 2*3.14);
    fill(current_array, 2, i_max/4, 0, 2*3.14);

    // Time & run the wave equation simulation in simulate.cc
    waveTimer.start();
    double *result_array = simulate(
        i_max, t_max, block_size,
        old_array, current_array, next_array
    );
    waveTimer.stop();

    // Print the time it took and write the result to a result.txt
    cout << waveTimer;
    file_write_double_array("result.txt", result_array, i_max);

    // Clean the arrays
    delete[] old_array;
    delete[] current_array;
    delete[] next_array;

    return EXIT_SUCCESS;
}
