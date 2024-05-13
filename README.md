# PARAFAC_T1

This is a method of performing trilinear analysis on large data sets using a modification of the PARAFAC-ALS algorithm.  It iteratively decomposes the data matrix into a core matrix and three loading matrices based on the Tucker1 model.  The algorithm is particularly useful for data sets that are too large to upload into a computer?s main memory.  While the performance advantage in utilizing our algorithm is dependent on the number of data elements and dimensions of the data array, we have seen a significant performance improvement over operating PARAFAC-ALS on the full data set.

SCR# 2471
