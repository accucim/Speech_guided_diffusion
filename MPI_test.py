from mpi4py import MPI
comm = MPI.COMM_WORLD

myrank = comm.Get_rank()
rank_size = comm.Get_size()

print("myrank is ", myrank)
print("rank size is ", rank_size)

# mpirun -np 2 python MPI_test.py
# mpiexec -n 2 python MPI_test.py