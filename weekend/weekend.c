//  weekend.c
//
//  Created by Alex Birch on 07/12/2013.
//  Copyright (c) 2013 Alex Birch. All rights reserved.
//	Some code re-used from previous assignment (credit Alex Birch)

#define MAXITERATIONS 100000000

/* Toggle verbose logging:
 * - prints matrix on each iteration
 * - prints buffers before we send, receive messages
 * - duplicate some messages on all processors
 * helps with debug
 * intended for separate xterms
 */
//#define verbose

// use floats or doubles
#define VALCHOICE 0
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

#define STR_VALUE(arg)      #arg
#define FUNCTION_NAME(name) STR_VALUE(name)
#define VALNAME FUNCTION_NAME(VALTYPE)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

// return pointer to a VALTYPE matrix with magnitude 'mag'
VALTYPE** buildMatrix(int width, int height) {
	// allocate cleared memory for width * height values of type VALTYPE
    VALTYPE* values = calloc(height * width, sizeof(VALTYPE));
    if (!values) {
    	fprintf(stderr, "matrix value calloc failed!\n");
		return 0;
    }

    // allocate memory for pointers to 'height' rows of VALTYPEs
    VALTYPE** rows = malloc(height * sizeof(VALTYPE*));
    if (!rows) {
		fprintf(stderr, "matrix row malloc failed!\n");
		return 0;
	}

    // point each row to its insertion point in value array
    int i;
    for (i=0; i<height; i++) {
        rows[i] = values + i*width;
    }
    return rows;
}

// return pointer to an int matrix with magnitude 'mag'
int** buildIntMatrix(int width, int height) {
	// allocate cleared memory for width * height values of type int
    int* values = calloc(height * width, sizeof(int));
    if (!values) {
    	fprintf(stderr, "matrix value calloc failed!\n");
		return 0;
    }

    // allocate memory for pointers to 'height' rows of VALTYPEs
    int** rows = malloc(height * sizeof(int*));
    if (!rows) {
		fprintf(stderr, "matrix row malloc failed!\n");
		return 0;
	}

    // point each row to its insertion point in value array.
    int i;
    for (i=0; i<height; i++) {
        rows[i] = values + i*width;
    }
    return rows;
}

// free() memory for malloc()ed VALTYPE matrix
void freeMatrix(VALTYPE **mat) {
	// free values
	free(mat[0]);
	// free row pointers
    free(mat);
}

// free matrix pool
void freeMatrixPool(VALTYPE ***matrixArray, int matrixCount) {
	int i;
	// free matrix pool
	for (i = 0; i<matrixCount; i++) {
		freeMatrix(matrixArray[i]);
	}
}

/* Initiates this matrix segment such that
 * whole square matrix, 'mat', dimension 'mag',
 * has 1s over first row and column.
 * Other values assumed 0.
 */
void initMatrix(VALTYPE **mat, int columns, int rows, int firstSeg) {
	// set first element in each row to 1
	int i;
    for (i=0; i<rows; i++) {
        mat[i][0] = 1.0f;
    }

    // top segment contains row 0, which must be filled with 1s.
    if (firstSeg) {
    	for (i=0; i<columns; i++) {
    		mat[0][i] = 1.0f;
    	}
    }
}

/* Print matrix 'mat', with 'columns'x'rows'.
 * Vulnerable to other prints occurring mid-matrix.
 */
void printMatrix(VALTYPE** mat, int columns, int rows) {
    printf("\n");
    int i, j;
    for (i=0; i<rows; i++) {
        for (j=0; j<columns; j++) {
            // i is row, j is column
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Print '1D' matrix, breaking into rows every 'columns' elements.
 * Mainly for receipt of matrix via MPI message.
 * Vulnerable to other prints occurring mid-matrix.
 * Can be used on a 2D matrix if pointed to correct insertion point.
 */
void print1DMatrix(VALTYPE* matrix, int columns, int rows) {
    int i, j;
    for (i=0; i<rows; i++) {
        for (j=0; j<columns; j++) {
            // i is row, j is column
            printf("%f ", matrix[i*columns + j]);
        }
        printf("\n");
    }
}

// print a row of VALTYPEs, 'columns' wide.
void printRow(VALTYPE* row, int columns) {
    printf("\n");
    int j;
	for (j=0; j<columns; j++) {
		// i is row, j is column
		printf("%f ", row[j]);
	}
    printf("\n\n");
}

/* Main worker function
 * Iteratively relaxes grid of numbers. Thread state & identity is tied to
 * a struct of arguments.
 */
signed int relaxGrid(int mag, VALTYPE precision, int procs, int rank, int matrixCount) {
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

	/* Calculates the row a given process starts operations from.
	 * This is an absolute position in the shared grid.
	 * From the first valid row, add minimum rows per proc
	 * then add to that the number of procs that took remainder rows so far
	 */
#define rowStartForProc(a) \
	( firstOperableRow + (a)*rowsMin + min((a), extraRows) )

	// matrix row to start operations from (absolute position)
	const int startRow = rowStartForProc(rank);
	/* matrix row to end operations at (absolute position)
	 * formula provides gapless coverage up to next proc's start row
	 */
	const int endRow = rowStartForProc(rank+1)-1;

	// how many rows this proc writes to in the loop
	const int myOperableRows = (endRow - startRow) + 1;

	// first row we write to (relative to our memory)
	const int myFirstOperableRow = 1;
	// last row we write to (relative to our memory)
	const int myLastOperableRow = myFirstOperableRow + myOperableRows - 1;

	/* Calculate how many values we need to read from in the loop.
	 * This value includes the neighbours of those values we write to.
	 * It also decides how much memory we need to allocate for the grid.
	 */
	// how many rows this proc reads from
	const int myReadRows = myOperableRows + 2;
	// how many columns this proc reads from
	const int myReadColumns = mag;

	// first row we read from (relative to our memory)
	const int myFirstReadRow = 0;
	// last row we read from (relative to our memory)
	const int myLastReadRow = myReadRows-1;

	// have each proc print out its own arguments
	printf("[RANK%d\t] Start Row: %d\tEnd Row: %d\n",
			rank, startRow, endRow);
	// have one thread print out values common to all procs
	if (rank == 0) {
		printf("Operable Rows: %d\tMin Rows/Proc: %d\tRemainder Rows: %d\n",
				operableRows, rowsMin, extraRows);
	}
#ifdef verbose
	// have each proc print out values relative to its own memory
	printf("[RANK%d\t] (Rel) Writable Start Row: %d\tEnd Row: %d\n",
			rank, myFirstOperableRow, myLastOperableRow);
	printf("[RANK%d\t] (Rel) Readable Start Row: %d\tEnd Row: %d\n",
			rank, myFirstReadRow, myLastReadRow);
#endif

	/*==============================================================*/
	// Allocate matrix pool memory
	/*==============================================================*/

	// array of matrices
	VALTYPE **matrices[matrixCount];

	// initialise matrix pool
	int i;
	// build as many matrices as our cache demands; one used per iteration
	for (i = 0; i<matrixCount; i++) {
		// allocate memory for matrix
		matrices[i] = buildMatrix(mag, mag);
		if (!matrices[i]) {
			fprintf(stderr, "matrix init failed!\n");
			return 1;
		}
		// rank == 0 proc fills top row with 1s
		initMatrix(matrices[i], myReadColumns, myReadRows, rank == 0);
	}

	// array to track whether each cached matrix achieved relaxation
	int progressArray[matrixCount];

	// progress types
	typedef enum {
		UNFINISHED = 0,	// iteration has not been evaluated yet
		NORELAX,		// iteration evaluated, was not relaxed
		RELAXED			// iteration evaluated, was relaxed
	} PROGRESS_TYPE;

	/*==============================================================*/
	// Iteration declarations
	/*==============================================================*/

	// current iteration number
	int n = 0;
	// current iteration to mod of cache size
	int iterationModCacheSize = 0;

	// iteration number for winning iteration
	signed int winningIterationAbs = -1;
	// winning iteration to modulus of cache size (points to matrix used)
	signed int winningIterationMod = -1;

	/*==============================================================*/
	// Matrix pointer declarations
	/*==============================================================*/

	// array index of source and destination matrices, for reading and writing
	int sourceMatrix, destMatrix;
	// pointer to source matrix
	VALTYPE** source;
	// pointer to destination matrix
	VALTYPE** dest;

	/*==============================================================*/
	// Message passing declarations
	/*==============================================================*/

	// message types
	typedef enum {
		ROW_DATA = 0,	// shared row update from neighbour
		MATRIX_DATA		// matrix portion sent upon final merge
	} MESSAGE_TYPE;

	// message tag and sender rank, of
	MPI_Status   rowReceiptStatus;
	// request structures for 'immediate' sending/receiving rows of territory
	MPI_Request	send_start = NULL,	send_end = NULL,
				recv_start;
	// pointers to where matrices, start/end rows should be sent from/written to
	VALTYPE     *sendStartRowBuff,	*sendEndRowBuff,
				*recvEndRowBuff,	*recvStartRowBuff,
				*recvMatrixBuff,	*sendMatrixBuff;

	// size of one row will be dimension of matrix
	int	rowBuffSize = mag;

	/*==============================================================*/
	// Algorithm iteration loop
	/*==============================================================*/

	// relax the matrix iteratively, until fully relaxed (or iteration limit)
	for (n=0; n<MAXITERATIONS; n++) {
		printf("[RANK%d\t] beginning iteration %d.\n", rank, n);

		// work out current iteration to mod of cache size
		iterationModCacheSize = n % matrixCount;

		// cycle source and destination pointers to next matrix in our cache
		sourceMatrix = iterationModCacheSize;
		// destMatrix is 'next' index in circular cache of iterations
		destMatrix = (sourceMatrix + 1) % matrixCount;

		// point source and destination to their respective matrices from array
		source = matrices[sourceMatrix];
		dest = matrices[destMatrix];

		// whether all values in our territory are relaxed this iteration.
		int relaxed = 1;

		/* Iterate through all matrix values within this thread's jurisdiction.
		 * For each value, find average of all cardinal neighbours.
		 */
		for (i=myFirstOperableRow; i<=myLastOperableRow; i++) {
			int j;
			for (j=firstOperableColumn; j<= lastOperableColumn; j++) {
				// Original value of this cell (for comparison)
				VALTYPE previousValue = source[i][j];

				// Find sum of neighbours
				VALTYPE surroundingValues = source[i-1][j];
				surroundingValues		 += source[i+1][j];
				surroundingValues		 += source[i][j-1];
				surroundingValues		 += source[i][j+1];

				// Find average of neighbours
				VALTYPE newValue = surroundingValues/4;
				// Write this value into destination matrix
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

		// write to progress array whether this iteration achieved relaxation
		progressArray[iterationModCacheSize] = relaxed ? RELAXED : NORELAX;
		printf("[RANK%d\t] For iteration %d (%d modx), relaxed: %d\n",
										rank, n, iterationModCacheSize,
										progressArray[iterationModCacheSize]);

#ifdef verbose
		printMatrix(dest, myReadColumns, myReadRows);
#endif

		/*==========================================================*/
		// Send rows to neighbours
		/*==========================================================*/

		// if you have a 'previous rank', exchange the edge row you share
		if (rank > 0) {
#ifdef verbose
			printf("Sending our start row to previous rank:\n");
			printRow(dest[myFirstOperableRow], rowBuffSize);
#endif
			// first row we wrote to in this iteration, needs to be shared
			sendStartRowBuff = dest[myFirstOperableRow];

			// make sure we finished sending last iteration's row
			if (send_start) {
				// wait for buffer to be emptied (blocking)
				MPI_Wait(&send_start, &rowReceiptStatus);
			}
			// (start) sending our top written row, to the previous rank
			MPI_Isend(sendStartRowBuff,rowBuffSize,MPI_VALTYPE,
							rank-1,(int)ROW_DATA,MPI_COMM_WORLD,&send_start);


#ifdef verbose
			printf("Received data will overwrite these first row contents:\n");
			printRow(dest[myFirstReadRow], rowBuffSize);
#endif
			// first row we read this iteration, needs to be updated by neighbor
			recvStartRowBuff = dest[myFirstReadRow];

			// (start) receiving our top read row, from the previous rank
			// non-blocking, so we can setup exchange of other row in meantime.
			MPI_Irecv(recvStartRowBuff,rowBuffSize,MPI_VALTYPE,
						   rank - 1,(int)ROW_DATA,MPI_COMM_WORLD,&recv_start);
		}

		// if you have a 'next rank', exchange the edge row you share
		if (rank < procs - 1) {
#ifdef verbose
			printf("Sending our end row to next rank:\n");
			printRow(dest[myLastOperableRow], rowBuffSize);
#endif
			// last row we wrote to in this iteration, needs to be shared
			sendEndRowBuff = dest[myLastOperableRow];

			// make sure we finished sending last iteration's row
			if (send_end) {
				// wait for buffer to be emptied (blocking)
				MPI_Wait(&send_end, &rowReceiptStatus);
			}
			// (start) sending our bottom written row, to the next rank
			MPI_Isend(sendEndRowBuff,rowBuffSize,MPI_VALTYPE,
							rank+1,(int)ROW_DATA,MPI_COMM_WORLD,&send_end);


#ifdef verbose
			printf("Received data will overwrite these last row contents:\n");
			printRow(dest[myLastReadRow], rowBuffSize);
#endif
			// last row we read this iteration, needs to be updated by neighbor
			recvEndRowBuff = dest[myLastReadRow];

			// receive our bottom read row, from next rank.
			// blocking; can't start next iteration without this information.
			MPI_Recv(recvEndRowBuff,rowBuffSize,MPI_VALTYPE,
						   rank + 1,(int)ROW_DATA,MPI_COMM_WORLD,&rowReceiptStatus);
		}

		// work we could do in the meantime has now been done; revisit this call
		if (rank > 0) {
			MPI_Wait(&recv_start, &rowReceiptStatus);
		}

#ifdef verbose
		printf("Final matrix for this iteration:\n");
		printMatrix(dest, myReadColumns, myReadRows);
#endif

		/*==========================================================*/
		// Check for completion criteria
		/*==========================================================*/

		/* If we have finished all iterations we can fit in our cache, we must
		 * check if any of them were winners before we overwrite the memory.
		 *
		 * Indexing is circular, so that the 'next' iteration can always be
		 * addressed as 'next' in memory. A resetting 0-to-n would also work.
		 *
		 * We do a fair bit of work to make sure we choose the 'first' winning
		 * iteration. This is to keep results deterministic, so that correctness
		 * testing is easier; all cache sizes will output same result.
		 * 		If we didn't care about this determinism, then we would only
		 * need to check the latest iteration (and no cache would be needed).
		 */
		if (n % matrixCount == matrixCount-1) {
#ifdef verbose
			printf("Gather!\n");
#endif

			// array to store min of each
			int relaxResults[matrixCount];
			// find minimum that everyone has for any given iteration (blocking)
			MPI_Allreduce(&progressArray, &relaxResults, matrixCount,
									MPI_INT, MPI_MIN, MPI_COMM_WORLD);

			// consider oldest cached iteration first
			// oldest will be the one immediately after newest, in modular.
			int oldest = (n + 1) % matrixCount;
			// ask all cached iterations: which was first to be wholly relaxed
			for (i=0; i<matrixCount; i++) {
				// check next newest matrix in cache (circular indexing)
				int currentMatrix = (oldest+i) % matrixCount;
#ifdef verbose
				printf("Checking iteration (mod): %d\n", i);
				printf("my relax on that iteration was: %d\n", progressArray[currentMatrix]);
#endif

#ifdef verbose
				printf("min is: %d\n", relaxResults[currentMatrix]);
#endif
				// if anyone is unrelaxed, min() will be < relaxed.
				if (relaxResults[currentMatrix] < RELAXED) {
					// clear progress history for that iteration, for reuse
					// not strictly necessary, but keeps status valid always.
					progressArray[currentMatrix] = UNFINISHED;
				} else {
					// finished; remember which index in array the matrix was.
					winningIterationMod = currentMatrix;
					// necessarily, winning iteration did not happen in future.

					// work out non-mod value for winning iteration.
					// abs difference will equal the mod difference
					winningIterationAbs = n -
							(iterationModCacheSize-winningIterationMod);

					// if (to mod) winning iteration is higher than current,
					// then it happened in the past, is actually lower.
					if (winningIterationMod > iterationModCacheSize) {
						// wind the iteration back by the modulus
						winningIterationAbs -= matrixCount;
					}

					// winner found; stop evaluating further matrices
					break;
				}
			}
			// If answer has been assigned,
			if (winningIterationMod >= 0) {
#ifdef verbose
				printf("******************\n");
#endif
				// do no further iterations
				break;
			}
		}
	}

	/*==============================================================*/
	// Catch erroneous exit
	/*==============================================================*/

	if (n == MAXITERATIONS) {
		printf("(MAXITERATIONS reached) proc %d is returning.\n", rank);

		// free heap memory
		freeMatrixPool(matrices, matrixCount);

		return EXIT_FAILURE;
	} else if (winningIterationMod < 0) {
		printf("(No winning iteration named) proc %d is returning.\n", rank);

		// free heap memory
		freeMatrixPool(matrices, matrixCount);

		return EXIT_FAILURE;
	}

	/*==============================================================*/
	// Give winning matrix to rank 0 to print
	/*==============================================================*/
	// find destMatrix used for the winning iteration
	int destMatrixMod = (winningIterationMod + 1) % matrixCount;
	// rank 0 receives matrix portions
	if (rank == 0) {
		printf("[RANK0] Wrapping up..\n");
		printf("First winning iteration was: %d\n", winningIterationAbs);
		printf("This is at matrix cache: %d\n", destMatrixMod);

		// matrix to print is the destMatrix written to by the winning iteration
		dest = matrices[destMatrixMod];
		// print top edge, plus all my operable rows
		int rank0Rows = 1 + myOperableRows;
		// final proc is responsible for bottom edge; if alone, would be rank 0
		if (0 == procs - 1) {
			rank0Rows++;
		}

		// print as 1D matrix indexed from first value
		// (just to treat it the same way as we'll treat proceeding messages)
		print1DMatrix(dest[0], myReadColumns, rank0Rows);

		// recycle my own matrix, since big enough, and no longer needed
		// big enough because rank 0 necessarily takes a remainder row if needed
		recvMatrixBuff = dest[0];

		// for some other process, how many rows it is responsible for
		int currentProcOwnedRows;
		// buffer size required to send that many rows
		int matrixBuffSize;

		// process number
		int p;
		// iterate through all processes, receiving their rows and printing them
		for (p=1; p<procs; p++) {
			// calculate start and end rows for that process
			int currentProcStartRow = rowStartForProc(p);
			int currentProcEndRow = rowStartForProc(p+1)-1;

			// calculate number of rows that process is responsible for
			currentProcOwnedRows = currentProcEndRow - currentProcStartRow + 1;

			// final proc is responsible for sending bottom edge
			if (p == procs - 1) {
				currentProcOwnedRows++;
			}

			// amount of data being received is number of rows * size of row
			matrixBuffSize = currentProcOwnedRows*rowBuffSize;
#ifdef verbose
			printf("Expecting buffer of size: %d\n", matrixBuffSize);
#endif

			// receive from proc p its portion of the matrix over our memory
			MPI_Recv(recvMatrixBuff, matrixBuffSize, MPI_VALTYPE,
					p, (int)MATRIX_DATA, MPI_COMM_WORLD, &rowReceiptStatus);

			// gaplessly, print that matrix portion next to the existing prints
			print1DMatrix(recvMatrixBuff, myReadColumns, currentProcOwnedRows);
		}
	} else {
		// ranks above 0 send matrix portions
#ifdef verbose
		printf("[RANK%d] Wrapping up..\n", rank);
		printf("First winning iteration was: %d\n", winningIterationAbs);
		printf("This is matrix: %d\n", destMatrixMod);
#endif

		// matrix to send is the destMatrix written to by the winning iteration
		dest = matrices[destMatrixMod];
		// send buffer starts from first row we are responsible for
		sendMatrixBuff = dest[myFirstOperableRow];

		// rows we are responsible for is at least the amount we write to
		int currentProcOwnedRows = myOperableRows;
		// final proc must also send a non-operable edge
		if (rank == procs - 1) {
			currentProcOwnedRows++;
		}

		// amount of data being received is number of rows * size of row
		int matrixBuffSize = rowBuffSize*currentProcOwnedRows;

#ifdef verbose
		printf("Sending buffer of size: %d\n", matrixBuffSize);
#endif

		// send to proc 0 our portion of the matrix from iteration cache
		MPI_Send(sendMatrixBuff,matrixBuffSize,MPI_VALTYPE,
						0,(int)MATRIX_DATA,MPI_COMM_WORLD);
	}

	/*==============================================================*/
	// Free heap memory
	/*==============================================================*/
	// free matrix pool
	freeMatrixPool(matrices, matrixCount);

	// return success; program completed its work
	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	// number of processes, our process number, number of iterations cached
	int size, rank, cacheSize;

	// Initialize MPI Execution Environment
	MPI_Init(&argc, &argv);

	// inquire for number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	// inquire which process number we are
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		printf("PROCS = %d\n",size);
	}

	// max useful cache size happens to be number of processes; this is default.
	cacheSize = size;

	// string describing the value type (float or double) used in our matrix
	const char *type = VALNAME;
	// magnitude of matrix
	int mag;
	// precision to relax matrix to
	VALTYPE prec;

	// First argument is executable name
	if (argc >= 3 && argc < 5) {
		int currentArg = 1;
		int parseMatches = 0;

		parseMatches = sscanf (argv[currentArg++], "%d", &mag);
		if (parseMatches != 1) {
			printf("Parse error on argument 0 [int gridSize]\n");

			MPI_Finalize();
			return EXIT_FAILURE;
		}
		// 65535*65535 matrix is largest we can index through using an int.
		if (mag>65535 || mag < 3) {
			printf("Boundary error on argument 0 [int 3~65535]; saw %d\n", mag);

			MPI_Finalize();
			return EXIT_FAILURE;
		}


	#if VALCHOICE == 0
		parseMatches = sscanf (argv[currentArg++], "%f", &prec);
		if (parseMatches != 1) {
			printf("Parse error on argument 2 [%s prec]\n", type);

			MPI_Finalize();
			return EXIT_FAILURE;
		}
		if (prec>0.1f || prec <0.000001f ) {
			printf("Boundary error arg 2 [%s 0.000001~0.1]; saw %f\n",
					type, prec);

			MPI_Finalize();
			return EXIT_FAILURE;
		}
	#elif VALCHOICE == 1
		parseMatches = sscanf (argv[currentArg++], "%lf", &prec);
		if (parseMatches != 1) {
			printf("Parse error on argument 2 [%s prec]\n", type);

			MPI_Finalize();
			return EXIT_FAILURE;
		}
		if (prec>0.1lf || prec <0.00001lf) {
			printf("Boundary error arg 2 [%s 0.00001~0.1]; saw %lf\n",
					type, prec);

			MPI_Finalize();
			return EXIT_FAILURE;
		}
	#endif
		// if optional 'cache size' argument is present
		if (argc == 4) {
			parseMatches = sscanf (argv[currentArg++], "%d", &cacheSize);
			if (parseMatches != 1) {
				printf("Parse error on argument 0 [int cacheSize]\n");

				MPI_Finalize();
				return EXIT_FAILURE;
			}
			if (cacheSize > size || cacheSize < 2) {
				printf("Boundary error on argument 0 [int 2~procs]; saw %d\n", cacheSize);

				MPI_Finalize();
				return EXIT_FAILURE;
			}
		} else {
			cacheSize = max(2, size);
		}
	} else {
		printf("Saw %d arguments; expected 2 (+1 optional).\n", argc-1);
		printf("\nUsage:\n");
		printf("weekend [int gridSize] [%s prec] ([int cacheSize])\n", type);
		printf("\nAllowed boundaries:\n");
		printf("weekend [int 3~65535] [%s 0.000001~0.1] ([int 2~procs])\n",
				type);
		printf("\nExample usages:\n");
		printf("weekend 100 0.005\n");
		printf("weekend 100 0.005 2\n");

		MPI_Finalize();
		// when arguments are missing, no computation is expected; success
		return EXIT_SUCCESS;
	}

	if (size > mag - 2) {
		printf("Size (%d) exceeds writable matrix rows, mag-2 (%d). Exiting.\n",
				size, mag - 2);

		MPI_Finalize();
		return EXIT_FAILURE;
	}

	if (rank == 0) {
		printf("Cached iterations = %d\n", cacheSize);
	}

	int endVal = relaxGrid(mag, prec, size, rank, cacheSize);

#ifdef verbose
	printf("Reached MPI Finalize.\n");
#endif

	// Barrier all processes, terminate MPI Execution environment.
	MPI_Finalize();

#ifdef verbose
	printf("Reached end of main.\n");
#endif

	// one final flush of stdout and stderr (likely not strictly necessary)
	fflush(stdout);
	fflush(stderr);

	return endVal;
}
