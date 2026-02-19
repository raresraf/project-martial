/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem.
 * @details This program calculates the minimum number of moves required to move a single '1'
 * in a 5x5 matrix to the center position (3, 3). A move consists of swapping the '1'
 * with an adjacent (up, down, left, or right) '0'.
 *
 * @algorithm The solution finds the coordinates of the '1' by reading the input stream
 * character by character and tracking the current row and column. Once the '1' is located,
 * the program calculates the Manhattan distance between the '1's position and the center
 * of the matrix (3, 3). The Manhattan distance `|x1 - x2| + |y1 - y2|` corresponds
 * exactly to the minimum number of single-step moves required.
 *
 * @time_complexity O(1), as the matrix size is fixed at 5x5. The program scans at most 25 elements.
 * @space_complexity O(1), as the matrix is not stored in memory; only coordinate variables are used.
 */
#include <cstdio>
#include <cstdlib>

using namespace std;

int main() {
  char ch;
  // r: current row, c: current column. Both are 1-indexed.
  int r = 1, c = 1;

  /**
   * Block Logic: Read the input character by character to find the location of the '1'.
   * The loop increments the column and row counters as it consumes characters, effectively
   * traversing the 5x5 grid.
   * Invariant: `r` and `c` hold the 1-indexed coordinates of the character being processed.
   */
  while ((ch = getchar()) != '1') {
    if (ch == '0')
      c++;

    /* Better not try converting 1D to 2D */
    // This original comment highlights the traversal logic. When a row is
    // finished (5 elements processed), the counters reset for the next row.
    if (c == 6) {
      c = 1;
      r++;
    }
  }

  // Block Logic: Calculate the Manhattan distance to the center (3, 3).
  // The variable 'c' is reused to store the final result.
  c = abs(3 - c) + abs(3 - r);

  printf("%d\n", c);
  return 0;
}
