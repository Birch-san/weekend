//
//  yukkuri.c
//
//  Created by Alex Birch on 28/10/2013.
//  Copyright (c) 2013 Alex Birch. All rights reserved.
//

// reasonable upper limit for solving problem; help Aquila out
#define MAXITERATIONS 100000000

// whether to use floats (0) or doubles (1)
#define VALCHOICE 0
#if VALCHOICE == 0
// type for matrix values
#define VALTYPE float
// function used for calculating absolute of matrix value
#define ABS(a) fabsf(a)
#elif VALCHOICE == 1
// type for matrix values
#define VALTYPE double
// function used for calculating absolute of matrix value
#define ABS(a) fabs(a)
#endif

// enable verbose logging (matrix print on each iteration)
//#define verbose

// create string for value of VALTYPE
#define STR_VALUE(arg)      #arg
#define FUNCTION_NAME(name) STR_VALUE(name)
#define VALNAME FUNCTION_NAME(VALTYPE)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <math.h>

/* Apple don't implement barriers; optional in POSIX standard.
 * For the sake of development on my computer, I use David Cairns' barrier
 * implementation from StackOverflow
 * http://stackoverflow.com/questions/3640853/performance-test-sem-t-v-s-dispat
 * ch-semaphore-t-and-pthread-once-t-v-s-dispat
 *
 *** This code will not be used on Aquila, since __APPLE__ won't be defined.***
 */
#ifdef __APPLE__

#ifndef PTHREAD_BARRIER_H_
#define PTHREAD_BARRIER_H_

#include <errno.h>

typedef int pthread_barrierattr_t;
typedef struct
{
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int tripCount;
} pthread_barrier_t;


int pthread_barrier_init(pthread_barrier_t *barrier,
		const pthread_barrierattr_t *attr, unsigned int count)
{
    if(count == 0)
    {
        errno = EINVAL;
        return -1;
    }
    if(pthread_mutex_init(&barrier->mutex, 0) < 0)
    {
        return -1;
    }
    if(pthread_cond_init(&barrier->cond, 0) < 0)
    {
        pthread_mutex_destroy(&barrier->mutex);
        return -1;
    }
    barrier->tripCount = count;
    barrier->count = 0;

    return 0;
}

int pthread_barrier_destroy(pthread_barrier_t *barrier)
{
    pthread_cond_destroy(&barrier->cond);
    pthread_mutex_destroy(&barrier->mutex);
    return 0;
}

int pthread_barrier_wait(pthread_barrier_t *barrier)
{
    pthread_mutex_lock(&barrier->mutex);
    ++(barrier->count);
    if(barrier->count >= barrier->tripCount)
    {
        barrier->count = 0;
        pthread_cond_broadcast(&barrier->cond);
        pthread_mutex_unlock(&barrier->mutex);
        return 1;
    }
    else
    {
        pthread_cond_wait(&barrier->cond, &(barrier->mutex));
        pthread_mutex_unlock(&barrier->mutex);
        return 0;
    }
}

#endif // PTHREAD_BARRIER_H_
#endif // __APPLE__

/* Typesafe min() macro as per GCC docs:
 * http://gcc.gnu.org/onlinedocs/gcc-3.4.6/gcc/Typeof.html#Typeof
 */
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

// return pointer to a matrix with magnitude 'mag'.
VALTYPE** buildMatrix(int mag) {
	// allocate cleared memory for width * height values of type VALTYPE
    VALTYPE* values = calloc(mag * mag, sizeof(VALTYPE));
    if (!values) {
    	fprintf(stderr, "matrix value calloc failed!\n");
		return 0;
    }

    // allocate memory for pointers to 'height' rows of VALTYPEs
    VALTYPE** rows = malloc(mag * sizeof(VALTYPE*));
    if (!rows) {
		fprintf(stderr, "matrix row malloc failed!\n");
		return 0;
	}

    // point each row to some insertion point in value array.
    int i;
    for (i=0; i<mag; i++) {
        rows[i] = values + i*mag;
    }
    return rows;
}

// free() memory for malloc()ed matrix
void freeMatrix(VALTYPE **mat, int mag) {
	// free values
	free(mat[0]);
	// free row pointers
	free(mat);
}

/* 1s over first column and row of square matrix 'mat', dimension 'mag'.
 * Other values assumed 0.
 */
void initMatrix(VALTYPE **mat, int mag) {
	int i;
    for (i=0; i<mag; i++) {
        mat[i][0] = 1.0f;
        mat[0][i] = 1.0f;
    }
}

/* print matrix 'mat', dimension 'mag'
 * fast and simple, but vulnerable to other prints occurring mid-matrix.
 */
void printMatrix(VALTYPE** mat, int mag) {
    printf("\n");
    int i, j;
    for (i=0; i<mag; i++) {
        for (j=0; j<mag; j++) {
            // i is row, j is column
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* atomic matrix printer
 * prints whole matrix in one printf() call; can be used during other printing.
 */
void atomicPrintMatrix(VALTYPE** mat, int mag) {
	// string to be built prior to printing
	/* Each value is 9 chars (x.xxxxxx ), there are width*height of these.
	 * There are 'height' line breaks, plus two for whitespace.
	 * Terminated by one \0 character.
	 */
	char *outputstring = malloc((mag*mag*9 + (mag + 2) + 1)*sizeof(char));
	if (!outputstring) {
		fprintf(stderr, "output buffer malloc failed!\n");
	}

	// build string containing each value in matrix, row-by-row.
	int i, j;
	for (i=0; i<mag; i++) {
		for (j=0; j<mag; j++) {
			asprintf(&outputstring, "%s%f ", outputstring, mat[i][j]);
		}
		asprintf(&outputstring, "%s\n", outputstring);
	}

	// print string
    printf("\n%s\n", outputstring);

    free(outputstring);
}

// structure passed into thread to give relax function its parameters
typedef struct threadargs {
	// which thread this is
	int threadnum;
	// how many threads exist in this program
	int THREADS;

	// precision to relax up to
	VALTYPE precision;

	// matrix magnitude
	int mag;

	// current matrix to read from
	VALTYPE** source;
	// current matrix to write to
	VALTYPE** dest;

	// signifies when all threads have finished current iteration
	pthread_barrier_t *barrier;

	// signifies which threads have achieved relaxation
	int *relaxedFlagsSource;
	// additional relaxed array (access alternates between iterations)
	int *relaxedFlagsDest;
} threadargs;

/* Main worker function
 * Iteratively relaxes grid of numbers. Thread state & identity is tied to
 * a struct of arguments.
 */
VALTYPE **relaxGrid(threadargs *ts) {
	/* create aliases for values of commonly used variables
	 * more to reduce typing / improve clarity than to increase performance
	 */
	const int threadnum = ts->threadnum;
	const int THREADS = ts->THREADS;

	/* 0-indexed, which col/row in grid is first able to be operated on.
	 * We skip edges, as per coursework spec.
	 */
	const int firstValidColumn = 1;
	const int firstValidRow = 1;

	/* Last col/row to operate on
	 * Length -1 is final element, so -2 avoids the edge.
	 */
	const int lastValidColumn = ts->mag - 2;
	const int lastValidRow = ts->mag - 2;

	/* Matrix rows are divided between threads such that each thread gets
	 * at least their fair share (int division) of rows, but also that the
	 * first few threads take those rows that remain after division.
	 */

	// number of rows surrounded by values
	const int validRows = lastValidRow - firstValidRow + 1;
	// minimum number of rows to give to a thread
	const int rowsMin = validRows / THREADS;
	// remainder of rows to allocate
	const int extraRows = validRows % THREADS;

	/* Calculates the row a given thread starts operations from
	 * From the first valid row, add minimum rows per thread
	 * then add to that the number of threads that took remainder rows so far
	 */
#define rowStartForThread(a) \
	( firstValidRow + (a)*rowsMin + min((a), extraRows) )

	// matrix row to start operations from
	const int startRow = rowStartForThread(threadnum);
	/* matrix row to end operations at
	 * formula provides gapless coverage up to next thread's start row
	 */
	const int endRow = rowStartForThread(threadnum+1)-1;

#ifdef verbose
	// have each thread print out its own arguments
	printf("Thread: %d\tStart Row: %d\tEnd Row: %d\n",
			threadnum, startRow, endRow);
	// have one thread print out values common to all threads
	if (threadnum == 0) {
		printf("Operable Rows: %d\tMin Rows/Thread: %d\tRemainder Rows: %d\n",
				validRows, rowsMin, extraRows);
	}
#endif

	// anneal the matrix iteratively, until fully relaxed (or iteration limit)
	int n;
	for (n=0; n<MAXITERATIONS; n++) {
		printf("thread %d beginning iteration %d.\n", threadnum, n);

		/* Boolean, whether all numbers in jurisdiction have settled down to
		 * within precision.
		 */
		int relaxed = 1;
		int i,j;
		/* Iterate through all matrix values within this thread's jurisdiction.
		 * For each value, find average of all cardinal neighbours.
		 */
		VALTYPE surroundingValues;
		for (i=startRow; i<=endRow; i++) {
			for (j=firstValidColumn; j<= lastValidColumn; j++) {
				// Original value of this cell (for comparison).
				VALTYPE previousValue = ts->source[i][j];

				// Find sum of neighbours
				surroundingValues  = ts->source[i-1][j];
				surroundingValues += ts->source[i+1][j];
				surroundingValues += ts->source[i][j-1];
				surroundingValues += ts->source[i][j+1];

				// Find average of neighbours
				VALTYPE newValue = surroundingValues/4;
				// Write this value into destination matrix
				ts->dest[i][j] = newValue;

				// absolute difference between new and previous value
				VALTYPE delta = ABS(newValue - previousValue);

				// If all values so far have been suitably relaxed,
				if (relaxed) {
					// check that this one, too is relaxed.
					relaxed = delta < ts->precision;
				}
			}
		}

		// Write to globally-visible location whether we achieved relaxation
		ts->relaxedFlagsDest[threadnum] = relaxed;

		/* This barrier prevents threads from starting the next iteration until
		 * all threads have finished this current iteration.
		 */
		printf("thread %d waiting for all threads to finish iteration %d.\n",
				threadnum, n);
		pthread_barrier_wait(ts->barrier);

		// whether all threads achieved relaxation this iteration
		int allRelaxed = 1;
		int a;
		for (a = 0; a <THREADS; a++) {
			// if any single thread did not achieve relaxation, stop checking.
			if (!ts->relaxedFlagsDest[a]) {
				allRelaxed = 0;
				break;
			}
		}

		// provided all threads are relaxed, we can return
		if (allRelaxed) {
			printf("thread %d is returning, final iteration %d.\n",
					threadnum, n);
			/* return pointer to the final matrix written to.
			 * only one thread's return value is being observed
			 */
			return ts->dest;
		}

// printing is slow, so only enable for debug purposes.
#ifdef verbose
		// only need one thread to print the matrix each iteration
		if (threadnum==0) {
			// print matrix in one printf(). slow but safe.
			atomicPrintMatrix(ts->dest, ts->mag);
		}
#endif

		/* Separate 'read' and 'write' matrices are used to avoid writing
		 * to something that is being read from (because race conditions affect
		 * outcome). Upon ending any iteration, the written-to matrix becomes
		 * the more up-to-date worldview, thus we make it the source matrix
		 * for the next iteration, and begin overwriting the previous source.
		 */
		// swap source & dest matrix pointers, for next iteration.
	    VALTYPE **tempGrid;
		tempGrid = ts->source;
		ts->source = ts->dest;
		ts->dest = tempGrid;

		/* Before the iteration barrier, each thread writes whether they're
		 * relaxed. After the barrier, they read whether all threads were
		 * relaxed. To avoid race conditions where one thread could finish the
		 * next iteration and write a newer 'relaxed' status as other threads
		 * read the results, we make sure subsequent iterations write to a
		 * different array, alternating between the two.
		 */
		// swap flag array pointers, for next iteration.
		int *tempRelaxedFlags = ts->relaxedFlagsDest;
		ts->relaxedFlagsDest = ts->relaxedFlagsSource;
		ts->relaxedFlagsSource = tempRelaxedFlags;
	}

	// declare that problem was never completed, and return null matrix pointer
	printf("(MAXITERATIONS reached) thread %d is returning.\n", threadnum);
	return NULL;
}

/* Assigns specified number of threads to work on relaxation problem of
 * specified grid magnitude, to specified precision.
 * Returns approximation of time spent working (minus final matrix print).
 */
double doParallelRelax(int mag, int THREADS, VALTYPE precision) {
    /* oldMatrix becomes first source matrix to read from
     * Initialize with 0ed matrix.
     */
    VALTYPE **oldMatrix = buildMatrix(mag);
    // ensure memory allocation succeeded
    if (!oldMatrix) {
    	fprintf(stderr, "matrix init failed!\n");
		return 0;
    }

    /* newMatrix becomes first destination matrix to write to
     * Initialize with 0ed matrix.
     */
    VALTYPE **newMatrix = buildMatrix(mag);
    // ensure memory allocation succeeded
    if (!newMatrix) {
		fprintf(stderr, "matrix init failed!\n");
		return 0;
	}

    // initialize these matrices' first column and row with 1s
    initMatrix(oldMatrix, mag);
    initMatrix(newMatrix, mag);

    // the curious can confirm that matrix reads as expected.
#ifdef verbose
    printf("Initial matrix:\n");
    printMatrix(oldMatrix, mag);
#endif


    // barrier to ensure everyone finishes iteration before starting next one
    pthread_barrier_t barrier;
    // initialize barrier
    if (pthread_barrier_init(&barrier, NULL, THREADS) != 0) {
    	fprintf(stderr, "iteration barrier init failed\n");
		return 0;
    }

    // signifies which threads have achieved relaxation
    int relaxedFlags[THREADS];
    // array for use on alternating iterations
    int relaxedFlagsAlt[THREADS];

    // pointer to whichever matrix becomes the final one
    VALTYPE **finalMatrix;

    // all but one thread will be pthreads. main thread joins in the fun also.
    pthread_t thread[THREADS-1];
    // for all threads working on problem, set of arguments declared.
    threadargs argsList[THREADS];

    // dole out work to each thread
    int i;
    for (i = 0; i<THREADS; i++) {
    	// which thread this is; thread numbering starts from main
    	argsList[i].threadnum = i;
    	// how many threads exist in this program
    	argsList[i].THREADS = THREADS;

    	// precision to relax up to
    	argsList[i].precision = precision;

    	// matrix magnitude
    	argsList[i].mag = mag;

    	// current matrix to read from
    	argsList[i].source = oldMatrix;
    	// current matrix to write to
    	argsList[i].dest = newMatrix;

    	// signifies when all threads have finished current iteration
    	argsList[i].barrier = &barrier;

    	// signifies which threads have achieved relaxation
    	argsList[i].relaxedFlagsSource = relaxedFlags;
    	// additional relaxed array (access alternates between iterations)
    	argsList[i].relaxedFlagsDest = relaxedFlagsAlt;

    	// final thread to call function is this thread
    	if (i==THREADS-1) {
    		/* final iteration of loop is ourself
    		 * we will listen to the return value of this one.
    		 */
    		finalMatrix = relaxGrid(&argsList[i]);
    	} else {
    		// on any other loop iteration, spawn a pthread and assign work
			if (pthread_create(&thread[i], NULL,
					(void *(*)(void*))relaxGrid, &argsList[i])) {
				// if pthread creation fails, abort everything.
				fprintf(stderr, "Failed to create thread!\n");
				return 0;
			}
    	}
    }

    // wait for all threads to finish their work before we end the program
    for (i = 0; i<THREADS-1; i++) {
    	pthread_join(thread[i], NULL);
    }

    // end time will be 0 unless problem successfully completed.
    double endTime = 0;

    // pointer to final matrix is returned if problem completes
    if (finalMatrix) {
    	// ascertain run time
    	endTime = clock();

    	// print final matrix
    	printf("Final matrix:\n");
    	printMatrix(finalMatrix, mag);
    } else {
    	// declare that problem wasn't solved.
    	printf("FAILURE; no pointer to matrix returned\n");
    }

    // free up memory malloc()ed for both matrices.
	freeMatrix(oldMatrix, mag);
	freeMatrix(newMatrix, mag);

	// destroy the barrier
	pthread_barrier_destroy(&barrier);

    return endTime;
}

// Threaded program to relax a matrix of specified magnitude to some precision.
int main(int argc, const char * argv[])
{
	// string describing the value type used in our matrix.
	const char *type = VALNAME;
	// magnitude of matrix
	int mag;
	// number of threads requested
	int threads;
	// precision to relax matrix to
	VALTYPE prec;

	/* Validate parameter count, and value (to reasonable boundaries).
	 * First argument is executable name; check for 4 args
	 */
	if (argc == 4) {
		int currentArg = 1;
		int parseMatches = 0;

		parseMatches = sscanf (argv[currentArg++], "%d", &mag);
		if (parseMatches != 1) {
			printf("Parse error on argument 0 [int gridSize]");
			return 0;
		}
		if (mag>100000 || mag < 3) {
			printf("Boundary error on argument 0 [int 3~100000]; saw %d", mag);
			return 0;
		}

		parseMatches = sscanf (argv[currentArg++], "%d", &threads);
		if (parseMatches != 1) {
			printf("Parse error on argument 1 [int threads]");
			return 0;
		}
		if (threads>16 || threads <1 ) {
			printf("Boundary error on argument 1 [int 1~16]; saw %d", threads);
			return 0;
		}
		// matrix rows doled out fairly between threads; must have enough rows!
		if (threads > mag-2) {
			printf("'Threads' exceeds operable rows in matrix.\n");
			printf("Grid must be a magnitude at least 2 greater than Threads.");
			return 0;
		}

		// parse precision as float or double depending on how compiled.
#if VALCHOICE == 0
		parseMatches = sscanf (argv[currentArg++], "%f", &prec);
		if (parseMatches != 1) {
			printf("Parse error on argument 2 [%s prec]", type);
		}
		if (prec>0.1 || prec <0.00001 ) {
			printf("Boundary error arg 2 [%s 0.00001~0.1]; saw %f", type, prec);
			return 0;
		}
#elif VALCHOICE == 1
		parseMatches = sscanf (argv[currentArg++], "%lf", &prec);
		if (parseMatches != 1) {
			printf("Parse error on argument 2 [%s prec]", type);
		}
		if (prec>0.1 || prec <0.00001 ) {
			printf("Boundary error arg 2 [%s 0.00001~0.1]; saw %lf",type, prec);
			return 0;
		}
#endif
	} else {
		// explain usage
		printf("Saw %d arguments; expected 3.\n", argc-1);
		printf("\nUsage:\n");
		printf("yukkuri [int gridSize] [int threads] [%s precision]\n", type);
		printf("\nAllowed boundaries:\n");
		printf("yukkuri [int 3~100000] [int 1~16] [%s 0.00001~0.1]\n", type);
		printf("\nExample usage:\n");
		printf("yukkuri 100 4 0.005.\n");

		return 0;
	}

	// print identifying characteristics of the run at the top of execution.
	printf("Threads:\t%d\n", threads);
	printf("Grid size:\t%d\n", mag);
#if VALCHOICE == 0
	printf("Precision:\t%f\n", prec);
#elif VALCHOICE == 1
	printf("Precision:\t%lf\n", prec);
#endif
	printf("Value type:\t%s\n\n", type);

	// Start timing the execution
	clock_t beginTime, endTime;
	double timeTaken;
	// current CPU time.
	beginTime = clock();

	/* dole out work across threads.
	 * time is returned upon successful completion of problem
	 */
	endTime = doParallelRelax(mag, threads, prec);

	if (endTime) {
		// (cumulative across all cores) elapsed CPU time
		timeTaken = (double)(endTime - beginTime) / CLOCKS_PER_SEC;

		printf("\nClock:\t\t%f\n\n", timeTaken);
	} else {
    	printf("Time disregarded; did not finish.\n");
	}

	// print the run circumstances at the bottom also.
	printf("Threads:\t%d\n", threads);
	printf("Grid size:\t%d\n", mag);
#if VALCHOICE == 0
	printf("Precision:\t%f\n", prec);
#elif VALCHOICE == 1
	printf("Precision:\t%lf\n", prec);
#endif
	printf("Value type:\t%s\n", type);

	printf("\nReached end of main.\n");

	// one final flush of stdout and stderr (likely not strictly necessary)
	fflush(stdout);
	fflush(stderr);

	// return 0 if we completed the problem, otherwise return 1.
    return endTime ? 0 : 1;
}
