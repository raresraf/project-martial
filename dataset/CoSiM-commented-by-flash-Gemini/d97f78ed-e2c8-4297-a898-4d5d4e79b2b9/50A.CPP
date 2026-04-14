/**
 * @file 50A.CPP
 * @brief This program attempts to solve the "Domino Piling" problem by simulating
 *        a specific greedy domino placement strategy on an N x M board.
 *        The problem typically asks for the maximum number of 1x2 dominoes that
 *        can be placed on an N x M board without overlapping, which has a simpler
 *        mathematical solution of (N * M) / 2. This code implements a direct simulation.
 * Algorithm: This code employs a specific greedy placement strategy. It iterates
 *        through each cell of the board. For an empty cell, it first attempts
 *        to place a 1x2 horizontal domino to its right. If that's not possible,
 *        and if the cell is at the end of a row, it then attempts to place a 2x1
 *        vertical domino below it. This specific simulation fills a 2D array
 *        representing the board.
 * Time Complexity: O(N * M) due to iterating through each cell of the board.
 * Space Complexity: O(N * M) for storing the board state in a 2D array.
 */
#include <cstdio> // Required for standard input/output functions like fscanf and fprintf.

using namespace std;

int main() {
  int N, M; // N: number of rows, M: number of columns for the board.

  // Reads the dimensions of the board (N and M) from standard input.
  fscanf(stdin, "%d%d", &N, &M);
  // Declares a Variable Length Array (VLA) to represent the board.
  // A[i][j] = 0 means the cell is empty, 1 means it's occupied by a domino.
  // Note: VLAs are a C99 feature and a GNU extension in C++, not standard C++.
  int A[N][M];

  int i, j;   // Loop counters.
  int count = 0; // Counts the number of dominoes successfully placed.

  /**
   * Block Logic: Initializes all cells of the board to 0 (empty).
   * Invariant: After this loop, every cell A[i][j] is marked as empty.
   */
  for (i = 0; i < N; ++i) {
    for (j = 0; j < M; ++j) {
      A[i][j] = 0; // Mark cell as empty.
    }
  }

  /**
   * Block Logic: Iterates through the board to place dominoes using a greedy strategy.
   * For each empty cell, it first attempts to place a horizontal domino (1x2) to its right.
   * If not possible, and it's at the end of a row, it attempts to place a vertical domino (2x1) below it.
   */
  for (i = 0; i < N; ++i) {
    for (j = 0; j < M; ++j) {
      // Condition 1: Attempt to place a horizontal domino.
      // Checks if the current cell and the cell to its right are empty and within bounds.
      if (A[i][j] == 0 && j + 1 < M && A[i][j + 1] == 0) {
        ++count;                // Increment domino count.
        A[i][j] = A[i][j + 1] = 1; // Mark both cells as occupied.
      }
      // Condition 2: Attempt to place a vertical domino (unusual condition).
      // Checks if the current cell is empty, is the last cell in its row, and the cell directly below it is empty.
      // This logic appears to specifically handle vertical placements only when at the end of a row, which may not be optimal.
      else if (A[i][j] == 0 && j + 1 == M && i + 1 < N && A[i + 1][j] == 0) {
        ++count;                // Increment domino count.
        A[i][j] = A[i + 1][j] = 1; // Mark both cells as occupied.
      }
    }
  }
  // Prints the total number of dominoes successfully placed to standard output.
  fprintf(stdout, "%d", count);
  // Standard program exit.
  return 0;
}
