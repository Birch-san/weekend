//  weekend.c
//
//  Created by Alex Birch on 07/12/2013.
//  Copyright (c) 2013 Alex Birch. All rights reserved.
//	Some code re-used from previous assignment (credit Alex Birch)

#define MAXITERATIONS 100000000

// use floats or doubles
#define VALCHOICE 1
#if VALCHOICE == 0
#define VALTYPE float
#define MPI_VALTYPE MPI_FLOAT
// function used for calculating absolute of matrix value
#define ABS(a) fabsf(a)
#elif VALCHOICE == 1
#define VALTYPE double
#define MPI_VALTYPE MPI_DOUBLE
// function used for calculating absolute of matrix value
#define ABS(a) fabs(a)
#endif

// enable verbose logging (matrix print on each iteration)
#define verbose

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define STR_VALUE(arg)      #arg
#define FUNCTION_NAME(name) STR_VALUE(name)
#define VALNAME FUNCTION_NAME(VALTYPE)

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

VALTYPE** buildMatrix(int width, int height) {
    VALTYPE* values = calloc(height * width, sizeof(VALTYPE));
    if (!values) {
    	fprintf(stderr, "matrix value calloc failed!\n");
		return 0;
    }

    VALTYPE** rows = malloc(height * sizeof(VALTYPE*));
    if (!rows) {
		fprintf(stderr, "matrix row malloc failed!\n");
		return 0;
	}

    int i;
    for (i=0; i<height; i++) {
        rows[i] = values + i*width;
    }
    return rows;
}

int** buildIntMatrix(int width, int height) {
    int* values = calloc(height * width, sizeof(int));
    if (!values) {
    	fprintf(stderr, "matrix value calloc failed!\n");
		return 0;
    }

    int** rows = malloc(height * sizeof(int*));
    if (!rows) {
		fprintf(stderr, "matrix row malloc failed!\n");
		return 0;
	}

    int i;
    for (i=0; i<height; i++) {
        rows[i] = values + i*width;
    }
    return rows;
}

void freeMatrix(VALTYPE **mat, int height) {
	free(mat[0]);
    free(mat);
}

void freeIntMatrix(int **mat, int height) {
	free(mat[0]);
    free(mat);
}

void initSquareMatrix(VALTYPE **mat, int mag, int firstSeg) {
	int i;
    for (i=0; i<mag; i++) {
        mat[i][0] = 1.0f;
        if (firstSeg) {
        	mat[0][i] = 1.0f;
        }
    }
}

void initMatrix(VALTYPE **mat, int columns, int rows, int firstSeg) {
	int i;
    for (i=0; i<rows; i++) {
        mat[i][0] = 1.0f;
    }
    if (firstSeg) {
    	for (i=0; i<columns; i++) {
    		mat[0][i] = 1.0f;
    	}
    }
}

void printMatrix(VALTYPE** mat, int columns, int rows) {
    printf("\n");
    for (int i=0; i<rows; i++) {
        for (int j=0; j<columns; j++) {
            // i is row, j is column
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printRow(VALTYPE* row, int columns) {
    printf("\n");
	for (int j=0; j<columns; j++) {
		// i is row, j is column
		printf("%f ", row[j]);
	}
    printf("\n\n");
}

typedef struct {
	// array of matrices
	VALTYPE*** matrices;
	// count of matrices in array
	int matrixCount;
	// matrix magnitude
	int mag;
	VALTYPE precision;
	// processor rank
	int rank;
	// number of processors
	int procs;
	// global variable saying whether threads are reaching final iteration
	int *wrapUp;
	// matrix of threads' progress on each iteration.
	int **threadProgressMatrix;
} identity;

signed int relaxGrid(int mag, VALTYPE precision, int procs, int rank) {
	/*==============================================================*/
	// Calculate row allocations for this proc
	/*==============================================================*/
	
	/* 0-indexed, which col/row in grid is first able to be operated on.
	 * We skip edges, as per coursework spec.
	 */
	const int firstOperableColumn = 1;
	const int firstOperableRow = 1;

	/* Last col/row to operate on
	 * Length -1 is final element, so -2 avoids the edge.
	 */
	const int lastOperableColumn = mag - 2;
	const int lastOperableRow = mag - 2;

	/* Matrix rows are divided between threads such that each proc gets
	 * at least their fair share (int division) of rows, but also that the
	 * first few procs take those rows that remain after division.
	 */

	// number of rows surrounded by values
	const int operableRows = lastOperableRow - firstOperableRow + 1;
	// minimum number of rows to give to a proc
	const int rowsMin = operableRows / procs;
	// remainder of rows to allocate
	const int extraRows = operableRows % procs;

	/* Calculates the row a given thread starts operations from
	 * From the first valid row, add minimum rows per proc
	 * then add to that the number of threads that took remainder rows so far
	 */
#define rowStartForProc(a) \
	( firstOperableRow + (a)*rowsMin + min((a), extraRows) )

	// matrix row to start operations from
	const int startRow = rowStartForProc(rank);
	/* matrix row to end operations at
	 * formula provides gapless coverage up to next proc's start row
	 */
	const int endRow = rowStartForProc(rank+1)-1;
	// how many rows this proc operates on
	const int myOperableRows = endRow - startRow;
	// how many rows this proc reads from
	const int myReadRows = myOperableRows + 2;
	// how many columns this proc reads from
	const int myReadColumns = mag;

#ifdef verbose
	// have each proc print out its own arguments
	printf("Rank: %d\tStart Row: %d\tEnd Row: %d\n",
			rank, startRow, endRow);
	// have one thread print out values common to all procs
	if (rank == 0) {
		printf("Operable Rows: %d\tMin Rows/Proc: %d\tRemainder Rows: %d\n",
				operableRows, rowsMin, extraRows);
	}
#endif
	
	/*==============================================================*/
	// Allocate matrix pool memory
	/*==============================================================*/
	// how many matrices will be in pool (iterations we can think ahead)
	const int matrixCount = max(2, procs+1);

	VALTYPE **matrices[matrixCount];

	// initialise matrix pool
	int i;
	for (i = 0; i<matrixCount; i++) {
		matrices[i] = buildMatrix(mag, mag);
		if (!matrices[i]) {
			fprintf(stderr, "matrix init failed!\n");
			return 0;
		}
		// rank == 0 proc fills top row with 1s
		initMatrix(matrices[i], myReadColumns, myReadRows, rank == 0);
	}

	// initialise progress tracking matrix
	int **progressMatrix;
	progressMatrix = buildIntMatrix(procs, matrixCount);
	int j;
	// 1 out the first row, since the built matrix is a complete iteration.
	for (j=0; j<procs; j++) {
		progressMatrix[0][j] = 1;
	}
	
	/*==============================================================*/
	// Algorithm
	/*==============================================================*/

	int sourceMatrix = 0;
	int destMatrix = sourceMatrix + 1;
	VALTYPE** source;
	VALTYPE** dest;

	int n;
	n = 0;

	printf("thread %d beginning iteration %d.\n", rank, n);
	source = matrices[sourceMatrix];
	dest = matrices[destMatrix];

	int relaxed = 1;
	VALTYPE surroundingValues;
	for (i=1; i<=myOperableRows; i++) {
		for (j=firstOperableColumn; j<= lastOperableColumn; j++) {
			VALTYPE previousValue = source[i][j];
			surroundingValues = 0;
			surroundingValues += source[i-1][j];
			surroundingValues += source[i+1][j];
			surroundingValues += source[i][j-1];
			surroundingValues += source[i][j+1];

			VALTYPE newValue = surroundingValues/4;
			dest[i][j] = newValue;

			// absolute difference between new and previous value
			VALTYPE delta = ABS(newValue - previousValue);

			// If all values so far have been suitably relaxed,
			if (relaxed) {
				// check that this one, too is relaxed.
				relaxed = delta < precision;
			}
		}
	}

	printMatrix(source, myReadColumns, myReadRows);

	/*int          taskid, ntasks;
	MPI_Status   status;
	MPI_Request	send_request,recv_request;
	int          ierr,i,j,itask,recvtaskid;
	int	        buffsize;
	double       *sendbuff,*recvbuff;
	double       sendbuffsum,recvbuffsum;
	double       sendbuffsums[1024],recvbuffsums[1024];
	double       inittime,totaltime,recvtime,recvtimes[1024];*/

	int          ntasks;
	MPI_Request	send_request,recv_request;
	MPI_Status   status;
	int          itask,recvtaskid;
	int	        buffsize;
	VALTYPE       *sendbuff,*recvbuff;

	buffsize = mag;

	if (rank > 0) {
		// row 0 was from someone else, and we do not edit it
		//printf("Sending our start row to previous rank:\n");
		//printRow(dest[1], buffsize);
		sendbuff = dest[1];
		// send start row to previous rank
		MPI_Isend(sendbuff,buffsize,MPI_VALTYPE,
						rank-1,0,MPI_COMM_WORLD,&send_request);
	}
	// send end row to next rank (instant)

	// now grab end row from next rank (blocking?)
	if (rank < procs - 1) {
		//printf("Current receiving row:\n");
		//printRow(dest[myReadRows-1], buffsize);
		recvbuff = dest[myReadRows-1];
		MPI_Irecv(recvbuff,buffsize,MPI_VALTYPE,
					   rank + 1,MPI_ANY_TAG,MPI_COMM_WORLD,&recv_request);
	}
	// and start row from previous rank (blocking?)

	//MPI_Wait(&send_request,&status);
	if (rank < procs - 1) {
		MPI_Wait(&recv_request, &status);
	}

	printMatrix(dest, myReadColumns, myReadRows);

	sourceMatrix = (sourceMatrix+1)%matrixCount;
	destMatrix = (destMatrix+1)%matrixCount;

	source = matrices[sourceMatrix];
	dest = matrices[destMatrix];



	/*==============================================================*/
	// Free matrix memory
	/*==============================================================*/
	
	// free matrix pool
	for (i = 0; i<matrixCount; i++) {
		freeMatrix(matrices[i], mag);
	}
	// free progress matrix
	freeIntMatrix(progressMatrix, matrixCount);

	/*==============================================================*/
	// Return
	/*==============================================================*/
	
	//printf("(MAXITERATIONS reached) proc %d is returning.\n", id->rank);
	printf("(EOF reached) proc %d is returning.\n", rank);
	return -1;
}

int doParallelRelax(int mag, VALTYPE precision, int procs, int rank) {
	relaxGrid(mag, precision, procs, rank);

	// success
	return 0;
}

int main(int argc, char **argv)
{
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("SIZE = %d RANK = %d\n",size,rank);

	const char *type = VALNAME;
	int mag;
	VALTYPE prec;

	// First argument is executable name
	if (argc == 3) {
		int currentArg = 1;
		int parseMatches = 0;

		parseMatches = sscanf (argv[currentArg++], "%d", &mag);
		if (parseMatches != 1) {
			printf("Parse error on argument 0 [int gridSize]\n");

			MPI_Finalize();
			return 0;
		}
		if (mag>1000 || mag < 3) {
			printf("Boundary error on argument 0 [int 3~1000]; saw %d\n", mag);

			MPI_Finalize();
			return 0;
		}


	#if VALCHOICE == 0
		parseMatches = sscanf (argv[currentArg++], "%f", &prec);
		if (parseMatches != 1) {
			printf("Parse error on argument 2 [%s prec]\n", type);

			MPI_Finalize();
			return 0;
		}
		if (prec>0.1 || prec <0.00001 ) {
			printf("Boundary error arg 2 [%s 0.00001~0.1]; saw %f\n",
					type, prec);

			MPI_Finalize();
			return 0;
		}
	#elif VALCHOICE == 1
		parseMatches = sscanf (argv[currentArg++], "%lf", &prec);
		if (parseMatches != 1) {
			printf("Parse error on argument 2 [%s prec]\n", type);

			MPI_Finalize();
			return 0;
		}
		if (prec>0.1 || prec <0.00001 ) {
			printf("Boundary error arg 2 [%s 0.00001~0.1]; saw %lf\n",
					type, prec);

			MPI_Finalize();
			return 0;
		}
	#endif
	} else {
		printf("Saw %d arguments; expected 2.\n", argc-1);
		printf("\nUsage:\n");
		printf("yukkuri [int gridSize] [int threads] [%s precision]\n", type);
		printf("\nAllowed boundaries:\n");
		printf("yukkuri [int 3~1000] [int 1~16] [%s 0.00001~0.1]\n", type);
		printf("\nExample usage:\n");
		printf("yukkuri 100 4 0.005.\n");

		MPI_Finalize();
		return 1;
	}

	int endVal = doParallelRelax(mag, prec, size, rank);

	printf("Reached MPI Finalize.\n");

	MPI_Finalize();

	printf("Reached end of main.\n");
	return(endVal);
}
