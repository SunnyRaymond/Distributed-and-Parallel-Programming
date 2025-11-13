/*
 * file.hh
 * 
 * Contains several functions for file I/O.
 * 
 * NOTE: YOU SHOULD NOT CHANGE THIS FILE
 * 
 */

#ifndef FILE_HH
#define FILE_HH

/* Returns the file size (in bytes) of the original.data file. */
int fileSize(const char* fileName);

/* Reads data from the given file. */
int readData(const char *fileName, char *data);

/* Writes data to the given file. */
int writeData(int size, const char *fileName, char *data);


#endif
