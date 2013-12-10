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

void print1DMatrix(VALTYPE* matrix, int columns, int rows) {
    for (int i=0; i<rows; i++) {
        for (int j=0; j<columns; j++) {
            // i is row, j is column
            printf("%f ", matrix[i*columns + j]);
        }
        printf("\n");
    }
}

void printRow(VALTYPE* row, int columns) {
    printf("\n");
	for (int j=0; j<columns; j++) {
		// i is row, j is column
		printf("%f ", row[j]);
	}
    printf("\n\n");
}

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
	// Declare message constants
	/*==============================================================*/
	typedef enum {
		ROW_DATA = 0,
		MATRIX_DATA,
		RELAX_UPDATE
	} MESSAGE_TYPE;

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

	// initialise progress tracking array
	int progressArray[matrixCount];
	// first iteration marked complete, unrelaxed
	progressArray[0] = 1;

	/*==============================================================*/
	// Algorithm variable declarations
	/*==============================================================*/
	signed int latestCheckedIteration = -1;
	// iteration number
	int n = 0;

	int sourceMatrix, destMatrix;
	VALTYPE** source;
	VALTYPE** dest;

	int relaxed;
	int j;
	/*==============================================================*/
	// Algorithm
	/*==============================================================*/
		MPI_Status   status;
		MPI_Request	send_start = NULL,send_end = NULL,
					recv_end,recv_start;
		VALTYPE       *sendStartRowBuff, *sendEndRowBuff,
						*recvEndRowBuff, *recvStartRowBuff,
						*recvMatrixBuff, *sendMatrixBuff;

		int	        rowBuffSize;
		VALTYPE surroundingValues;

		int min;
				int max;

	for (n=0; n<1; n++) {
		sourceMatrix = n % matrixCount;
		destMatrix = (n + 1) % matrixCount;

		printf("thread %d beginning iteration %d.\n", rank, n);
		source = matrices[sourceMatrix];
		dest = matrices[destMatrix];

		relaxed = 1;
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

		progressArray[destMatrix] = relaxed ? 2 : 1;

		printMatrix(dest, myReadColumns, myReadRows);

		/*==============================================================*/
		// Send rows to neighbours
		/*==============================================================*/


		/* btw we could totally send these messages as soon as we calculate
		 * the relevant row
		 */
		rowBuffSize = mag;

		// send start row to previous rank
		if (rank > 0) {
			// row 0 was from someone else, and we do not edit it
			printf("Sending our start row to previous rank:\n");
			printRow(dest[1], rowBuffSize);
			sendStartRowBuff = dest[1];
			// make sure we finished sending any existing request
			if (send_start) {
				MPI_Wait(&send_start, &status);
			}
			MPI_Isend(sendStartRowBuff,rowBuffSize,MPI_VALTYPE,
							rank-1,(int)ROW_DATA,MPI_COMM_WORLD,&send_start);
		}
		// send end row to next rank (instant)
		if (rank < procs - 1) {
			// row end-1 was from someone else, and we do not edit it
			printf("Sending our end row to next rank:\n");
			printRow(dest[myReadRows-2], rowBuffSize);
			sendEndRowBuff = dest[myReadRows-2];
			// make sure we finished sending any existing request.
			if (send_end) {
				MPI_Wait(&send_end, &status);
			}
			MPI_Isend(sendEndRowBuff,rowBuffSize,MPI_VALTYPE,
							rank+1,(int)ROW_DATA,MPI_COMM_WORLD,&send_end);
		}

		// now grab end row from next rank (blocking?)
		if (rank < procs - 1) {
			//printf("Current receiving row:\n");
			//printRow(dest[myReadRows-1], buffsize);
			recvEndRowBuff = dest[myReadRows-1];
			MPI_Irecv(recvEndRowBuff,rowBuffSize,MPI_VALTYPE,
						   rank + 1,(int)ROW_DATA,MPI_COMM_WORLD,&recv_end);
		}
		// grab start row from previous rank (blocking?)
		if (rank > 0) {
			//printf("Current receiving row:\n");
			//printRow(dest[0], buffsize);
			recvStartRowBuff = dest[0];
			MPI_Irecv(recvStartRowBuff,rowBuffSize,MPI_VALTYPE,
						   rank - 1,(int)ROW_DATA,MPI_COMM_WORLD,&recv_start);
		}

		if (rank < procs - 1) {
			MPI_Wait(&recv_end, &status);
		}
		if (rank > 0) {
			MPI_Wait(&recv_start, &status);
		}

		printMatrix(dest, myReadColumns, myReadRows);

		/*==============================================================*/
		// Read everyone's relaxed
		/*==============================================================*/

		/* the person who does this should be the person who first hits
		 * matrixCount above latestCheckedIteration
		 * although.. with reduce, everyone needs to do at same time.
		 */

		/* This is basically a barrier every iteration.. actually nobody will ever
		 * be 0, because they all necessarily got through their code if they are
		 * here.
		 */

		/* How about doing it every matrixCount iterations (so we reduce as much
		 * work as possible)?
		 * Although, pretty much guarantees you will have 0s
		 * Maybe do matrixCount of these every matrixCount iterations?
		 * Although only one person can be matrixCount ahead, so everyone else is 0
		 */

		/* Every matrixCount iterations, you declare the highest relax you have.
		 * If everyone has a 'relaxed' somewhere, we investigate. Uh
		 */

		/* Doesn't matter if everyone has finished an iteration if we can show that
		 * that iteration is not relaxed.
		 * Every matrixCount iterations, ask what the highest value for each
		 * iteration was. If there is a 1 ('finished, not relaxed'), increment.
		 * If max is 2, then check that min is also 2.
		 */

		/* If I'm relaxed, do I need to ask, or just tell I've signed off?*/
		/* Since this is also being used to check that everyone's done
		 * with the iteration.. but then again, only I need to be done with it.
		 */
		/* The reason the limit exists is so that you necessarily still have the
		 * data for that iteration in your buffer. Guess it doesn't matter if you
		 * overwrite an iteration that can't win. But you won't overwrite anyway?
		 * You can only cycle after if I have read at least my value.
		 */

		printf("Gather!\n");
		int nextIterationCheck = (latestCheckedIteration + 1) % matrixCount;
		printf("progressArray is: %d\n", progressArray[nextIterationCheck]);

		//int *relaxResult;
		int relaxResults[procs];
		//int *relaxResults = calloc(procs, sizeof(int));
	//	if (rank == 0) {
			//if (relaxed) {

		// looks like my mistake was expecting 'procs' receive count
		// receive count is actually 'per proc'
		MPI_Allgather(&progressArray[nextIterationCheck], 1, MPI_INT,
					   relaxResults, 1, MPI_INT, MPI_COMM_WORLD);

		// looks like both results go into element 0 of relaxResults?


		MPI_Reduce_local(relaxResults, &min, procs, MPI_INT, MPI_MIN);
		MPI_Reduce_local(relaxResults, &max, procs, MPI_INT, MPI_MAX);
		printf("Min was: %d \tMax was: %d\n", min, max);

		printf("progressArray is: %d\n", progressArray[nextIterationCheck]);
			//}
	//	}

		/*
		if (relaxResult == 0) {

		} else if (relaxResult == 1) {

		} else {

		}*/

		/*==============================================================*/
		// Swap pointers
		/*==============================================================*/

		/*sourceMatrix = (sourceMatrix+1)%matrixCount;
		destMatrix = (destMatrix+1)%matrixCount;

		source = matrices[sourceMatrix];
		dest = matrices[destMatrix];*/

		/*==============================================================*/
		// Give winning matrix to rank 0 to print
		/*==============================================================*/
		if (rank == 0) {
			printf("Wrapping up..\n");

			// print all my operable rows, plus top edge
			print1DMatrix(dest[0], myReadColumns, myReadRows-1);

			// recycle my own matrix, since big enough, and no longer needed
			recvMatrixBuff = dest[0];
			int currentProcOwnedRows;
			int matrixBuffSize;

			int p;
			for (p=1; p<procs; p++) {
				int currentProcStartRow = rowStartForProc(p);
				int currentProcEndRow = rowStartForProc(p+1)-1;
				currentProcOwnedRows = currentProcEndRow - currentProcStartRow;

				// final proc is also responsible for sending non-operable edge
				if (p == procs - 1) {
					currentProcOwnedRows++;
				}

				matrixBuffSize = currentProcOwnedRows*rowBuffSize;

				//printf("Expecting buffer of size: %d\n", matrixBuffSize);

				MPI_Recv(recvMatrixBuff, matrixBuffSize, MPI_VALTYPE,
						p, (int)MATRIX_DATA, MPI_COMM_WORLD, &status);

				print1DMatrix(recvMatrixBuff, myReadColumns, currentProcOwnedRows);
			}
		} else {
			sendMatrixBuff = dest[1];
			int currentProcOwnedRows = myOperableRows;
			// final proc must also send a non-operable edge
			if (rank == procs - 1) {
				currentProcOwnedRows++;
			}

			int matrixBuffSize = rowBuffSize*currentProcOwnedRows;

			printf("Sending buffer of size: %d\n", matrixBuffSize);

			// send my operable values to rank 0
			MPI_Send(sendMatrixBuff,matrixBuffSize,MPI_VALTYPE,
							0,(int)MATRIX_DATA,MPI_COMM_WORLD);
		}
	}

	/*==============================================================*/
	// Free matrix memory
	/*==============================================================*/
	
	// free matrix pool
	/*for (i = 0; i<matrixCount; i++) {
		freeMatrix(matrices[i], mag);
	}*/

	/*==============================================================*/
	// Return
	/*==============================================================*/
	
	//printf("(MAXITERATIONS reached) proc %d is returning.\n", id->rank);
	printf("(EOF1 reached) proc %d is returning.\n", rank);
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
