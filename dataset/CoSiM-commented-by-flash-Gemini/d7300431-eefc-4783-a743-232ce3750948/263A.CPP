/* You've got a 5   5 matrix, consisting of 24 zeroes and a single number one.
Let's index the matrix rows by numbers from 1 to 5 from top to bottom, let's
index the matrix columns by numbers from 1 to 5 from left to right. In one move,
you are allowed to apply one of the two following transformations to the matrix:

Swap two neighboring matrix rows, that is, rows with indexes i and i + 1 for
some integer i (1   i < 5). Swap two neighboring matrix columns, that is,
columns with indexes j and j + 1 for some integer j (1   j < 5). You think that
a matrix looks beautiful, if the single number one of the matrix is located in
its middle (in the cell that is on the intersection of the third row and the
third column). Count the minimum number of moves needed to make the matrix
beautiful.

Input
The input consists of five lines, each line contains five integers: the j-th
integer in the i-th line of the input represents the element of the matrix that
is located on the intersection of the i-th row and the j-th column. It is
guaranteed that the matrix consists of 24 zeroes and a single number one.

Output
Print a single integer   the minimum number of moves needed to make the matrix
beautiful. */

/**
 * @file 263A.CPP
 * @brief Solution to a competitive programming problem: "Beautiful Matrix".
 *
 * This program calculates the minimum number of moves required to position
 * the single '1' in a 5x5 matrix to its center (row 3, column 3, 1-indexed,
 * or row 2, column 2, 0-indexed). A move consists of swapping two neighboring
 * rows or columns.
 *
 * Algorithm: Manhattan distance calculation. The minimum number of moves is
 *            the sum of the absolute differences between the current row/column
 *            of '1' and the target row/column (center).
 * Time Complexity: O(1) for a fixed 5x5 matrix (specifically O(M*N) for matrix traversal,
 *                  which becomes O(1) since M, N are constants here).
 * Space Complexity: O(1) for storing the 5x5 matrix.
 */
#include <algorithm> // Required for std::find and std::abs.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <iterator>  // Required for std::distance.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

/**
 * @brief Main function: entry point of the program.
 * Reads a 5x5 matrix, finds the position of the element '1', and then
 * calculates the minimum moves to bring it to the center (2,2).
 */
int main() {
  int mat[5][5]; // Declares a 5x5 integer matrix.
  int x, y;      // Stores the 0-indexed row (x) and column (y) of the element '1'.

  /**
   * @brief Reads the 5x5 matrix elements from standard input.
   * Block Logic: Nested loops iterate through each cell of the matrix,
   * reading an integer into `mat[i][j]`.
   * Pre-condition: Input consists of 5 lines, each with 5 integers.
   */
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++)
      cin >> mat[i][j];

  //---------------------------------------------//
  /**
   * @brief Block to find the 0-indexed coordinates of the element '1'.
   * This section uses C++11 features for iterating and finding an element
   * within the 2D array treated as a contiguous block of memory.
   *
   * To compile with C++11 support, typically use: `g++ <filename>.cpp -std=c++11`
   */
  auto begin = reinterpret_cast<int *>(mat); // Inline: Casts the 2D array to a pointer to its first element, treating it as a 1D array.
  auto end = reinterpret_cast<int *>(mat) + 25; // Inline: Calculates the end pointer for the 1D view of the array.
  auto ind = find(begin, end, 1); // Inline: Uses `std::find` to locate the element '1' in the 1D view.

  if (ind != end) { // Checks if '1' was found within the matrix.
    size_t n = distance(begin, ind); // Inline: Calculates the 1D index 'n' of the element '1'.
    x = n / 5; // Calculates the 0-indexed row 'x' from the 1D index.
    y = n % 5; // Calculates the 0-indexed column 'y' from the 1D index.
  }
  //---------------------------------------------//

  /**
   * @brief Calculates the total minimum moves using Manhattan distance.
   * Block Logic: The minimum number of moves is the sum of the absolute differences
   * between the current row (`x`) and the target row (2), and the current column (`y`)
   * and the target column (2).
   * Pre-condition: `x` and `y` hold the 0-indexed coordinates of '1'.
   */
  int tot_moves = abs(x - 2) + abs(y - 2); // Target center is (2, 2) for 0-indexed matrix.
  cout << tot_moves << endl;              // Prints the calculated total moves.
  return 0;                               // Indicates successful program execution.
}