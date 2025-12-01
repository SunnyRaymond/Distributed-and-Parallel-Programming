/* mympi_bcast.h
 *
 * Declaration of MYMPI_Bcast: a broadcast implemented using
 * point-to-point communication on a 1D ring.
 */

#ifndef MYMPI_BCAST_H
#define MYMPI_BCAST_H

#include <mpi.h>

int MYMPI_Bcast(void *buffer,
                int count,
                MPI_Datatype datatype,
                int root,
                MPI_Comm communicator);

#endif /* MYMPI_BCAST_H */
