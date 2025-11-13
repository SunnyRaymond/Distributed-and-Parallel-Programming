/*
 * File.cc
 * 
 * Contains several functions for file I/O.
 * 
 * NOTE: YOU SHOULD NOT CHANGE THIS FILE
 * 
 */

#include <fstream>
#include <iostream>

#include "file.hh"

using namespace std;


/* Write array to file */
void file_write_double_array(const char *filename, double *array, int n) {
    ofstream myfile(filename);
    if (myfile.is_open()) {
        for(int count = 0; count < n; count++){
            myfile << array[count] << "\n";
        }
        myfile.close();
    }
    else {
        cout << "Unable to open file";
    }
}
