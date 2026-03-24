/**
 * @file 50A.CPP
 * @brief An attempted solution for Codeforces Problem 50A - "Domino piling".
 * @details This program attempts to solve the domino piling problem using a greedy
 * algorithm that simulates placing dominoes on a grid.
 *
 * @author (unknown)
 *
 * @algorithm
 * The program implements a flawed greedy strategy:
 * 1. It reads the dimensions of a board, N x M.
 * 2. It creates an N x M matrix `A` to represent the board, initialized to 0s (empty).
 * 3. It iterates through each cell `(i, j)` of the board.
 * 4. For each cell, it first tries to place a horizontal domino if the current cell
 *    and the cell to its right `(i, j+1)` are both empty.
 * 5. If a horizontal placement is not possible, it has a very specific check to place
 *    a vertical domino, but *only* if the current cell is in the last column (`j+1 == M`).
 * 6. When a domino is placed, the corresponding cells in the matrix are marked as 1 (filled),
 *    and a counter is incremented.
 * 7. The final count is printed.
 *
 * @note This greedy algorithm is incorrect and will not produce the optimal result for
 * many board sizes. The simple mathematical solution is `(N * M) / 2`. The code also uses
 * Variable Length Arrays (e.g., `A[N][M]`), which is a non-standard C++ extension
 * and is not guaranteed to compile on all compilers. It also uses C-style I/O (`fscanf`, `fprintf`).
 *
 * @complexity
 * Time Complexity: O(N * M) due to iterating through the entire grid.
 * Space Complexity: O(N * M) to store the grid.
 */
#include <cstdio>

using namespace std;

int main() {
  int N, M;

  // Using C-style file I/O to read from standard input.
  fscanf(stdin, "%d%d", &N, &M);
  // Using a Variable Length Array (VLA), a non-standard C++ feature.
  int A[N][M];

  int i, j, count = 0;

  // Initialize the board representation to all zeros (empty).
  for (i = 0; i < N; ++i) {
    for (j = 0; j < M; ++j) {
      A[i][j] = 0;
    }
  }

  /**
   * Block Logic: Implements a flawed greedy strategy to tile the board.
   * It iterates through each cell and tries to place a domino.
   */
  for (i = 0; i < N; ++i) {
    for (j = 0; j < M; ++j) {
      /**
       * Block Logic: Greedily places a horizontal domino if space is available.
       */
      if (A[i][j] == 0 && j + 1 < M && A[i][j + 1] == 0) {
        ++count;
        A[i][j] = A[i][j + 1] = 1; // Mark cells as occupied.
      } 
      /**
       * Block Logic: A very restrictive and incorrect condition to place a vertical domino.
       * This only considers placing a vertical domino in the last column of the board,
       * missing most opportunities.
       */
      else if (A[i][j] == 0 && j + 1 == M && i + 1 < N && A[i + 1][j] == 0) {
        ++count;
        A[i][j] = A[i + 1][j] = 1; // Mark cells as occupied.
      }
    }
  }
  
  // Using C-style file I/O to print the result to standard output.
  fprintf(stdout, "%d", count);
  return 0;
}
