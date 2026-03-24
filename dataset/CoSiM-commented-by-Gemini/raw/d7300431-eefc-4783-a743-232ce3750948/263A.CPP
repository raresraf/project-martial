/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A - Beautiful Matrix.
 * @details The problem asks for the minimum number of swaps of adjacent rows or columns
 * to move the single '1' in a 5x5 matrix to the center (row 3, column 3).
 * This is equivalent to calculating the Manhattan distance from the '1's current
 * position to the center.
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. A 5x5 matrix is read from standard input.
 * 2. The code then flattens the 2D matrix into a 1D array representation to use `std::find`.
 * 3. `std::find` is used to locate the element '1' in the flattened array.
 * 4. The 1D index of '1' is converted back to 2D coordinates (row `x`, column `y`).
 * 5. The Manhattan distance is calculated between the current coordinates (x, y) and the
 *    center coordinates (2, 2) using the formula: |x - 2| + |y - 2|.
 * 6. The result, which is the minimum number of moves, is printed to standard output.
 *
 * @complexity
 * Time Complexity: O(1), as the matrix size is fixed at 5x5 (25 elements). The read and find operations take constant time.
 * Space Complexity: O(1), for storing the 5x5 matrix.
 */
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

#include <algorithm>
#include <iostream>
#include <iterator>

using namespace std;

int main() {
  int mat[5][5];
  int x, y;
  /**
   * Block Logic: Reads the 5x5 matrix from standard input.
   */
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++)
      cin >> mat[i][j];
  //---------------------------------------------//
  /*
     This block contains code pertaining to C++11
     standards. To compile, type in terminal :
     g++ <filename>.cpp -std=c++11
  */
  /**
   * Block Logic: This block finds the coordinates of the '1' in the matrix.
   * It treats the 2D array `mat` as a contiguous 1D array of 25 integers.
   * `std::find` is used to get an iterator to the '1'.
   * The distance from the beginning of the array is then used to calculate the
   * 0-indexed row (x) and column (y) of the '1'.
   */
  auto begin = reinterpret_cast<int *>(mat);
  auto end = reinterpret_cast<int *>(mat) + 25;
  auto ind = find(begin, end, 1);
  if (ind != end) {
    size_t n = distance(begin, ind);
    x = n / 5; // Integer division gives the row index.
    y = n % 5;  // Modulo operation gives the column index.
  }
  //---------------------------------------------//
  /**
   * @brief Calculates the Manhattan distance from the '1's position (x, y) to the center (2, 2).
   * This distance represents the minimum number of row and column swaps required.
   */
  int tot_moves = abs(x - 2) + abs(y - 2);
  cout << tot_moves << endl;
  return 0;
}