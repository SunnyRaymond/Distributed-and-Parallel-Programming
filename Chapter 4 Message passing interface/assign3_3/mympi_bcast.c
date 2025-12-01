/* mympi_bcast.c
 *
 * Implementation of MYMPI_Bcast using point-to-point communication
 * on a 1D ring topology.
 */

#include <mpi.h>
#include "mympi_bcast.h"

int MYMPI_Bcast(void *buffer,
                int count,
                MPI_Datatype datatype,
                int root,
                MPI_Comm communicator)
{
    int rank, size;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);

    if (size == 1 || count == 0) {
        /* Nothing to do */
        return MPI_SUCCESS;
    }

    /* Ring pipeline:
     * step k: process (root + k) sends to (root + k + 1)
     * All indices taken modulo size.
     */
    MPI_Status status;
    int steps = size - 1;

    for (int k = 0; k < steps; ++k) {
        int sender   = (root + k)     % size;
        int receiver = (root + k + 1) % size;

        if (rank == sender) {
            /* send to next neighbour on the ring */
            MPI_Send(buffer, count, datatype, receiver, 0, communicator);
        } else if (rank == receiver) {
            /* receive from previous neighbour on the ring */
            MPI_Recv(buffer, count, datatype, sender, 0,
                     communicator, &status);
        }
        /* all other ranks are idle during this step */
    }

    return MPI_SUCCESS;
}
