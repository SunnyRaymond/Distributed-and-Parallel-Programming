/*
 * File.cc
 * 
 * Contains several functions for file I/O.
 * 
 * NOTE: YOU SHOULD NOT CHANGE THIS FILE
 * 
 */

#include <iostream>
#include <fstream>

#include "file.hh"

using namespace std;


/* Returns the file size (in bytes) of the original.data file. */
int fileSize(const char *fileName) {
  int size;

  ifstream file (fileName, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    file.close();
  }
  else {
    cout << "Unable to open file";
    size = -1;
  }
  return size;
}

/* Reads data from the given file. */
int readData(const char *fileName, char *data) {

  streampos size;

  ifstream file (fileName, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    file.seekg (0, ios::beg);
    file.read (data, size);
    file.close();

    cout << "The entire file content is in memory." << endl;
  }
  else cout << "Unable to open file" << endl;
  return 0;
}

/* Writes data to the given file. */
int writeData(int size, const char *fileName, char *data) {
  ofstream file (fileName, ios::out|ios::binary|ios::trunc);
  if (file.is_open())
  {
    file.write (data, size);
    file.close();

    cout << "The entire file content was written to file." << endl;
    return 0;
  }
  else cout << "Unable to open file";

  return -1;
}
