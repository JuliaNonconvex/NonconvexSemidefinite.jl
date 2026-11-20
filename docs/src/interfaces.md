## Matrix interfaces

For every `n*n` real positive semidefinite matrix that optimization objective contains, please have two inputs `x_L` and `x_D` representing the lower-triangular and the diagonal part of it. In the function, call `decompress_symmetric(x_L, x_d)` to represent that matrix, which will be handled by Nonocnvex automatically
