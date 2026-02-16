/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces (Problem 263A).
 *
 * This program reads a 5x5 matrix of integers. It identifies the row and column
 * indices of the element with value '1' and then calculates the minimum number
 * of moves required to bring this '1' to the center of the matrix (position [2][2]).
 * The number of moves is determined by the Manhattan distance to the center.
 */

#include <iostream> // @brief Provides input/output stream objects (cin, cout).
#include <stdlib.h> // @brief Provides general utilities, including the `abs` function.

using namespace std;

int main() {
  /**
   * @brief Main function to solve the Beautiful Matrix problem.
   *
   * This function reads a 5x5 integer matrix from standard input. It then
   * searches for the element with the value '1', stores its row (`l`) and
   * column (`k`) indices. Finally, it calculates and prints the minimum
   * number of moves required to bring this '1' to the center of the matrix,
   * which is at indices [2][2]. The calculation uses the Manhattan distance.
   *
   * @return int: 0 upon successful execution.
   */
  int mat[5][5], l, k; // @brief Declare a 5x5 integer matrix and variables for storing the coordinates of '1'.
  
  // @brief Loop through the 5x5 matrix to read input and find the position of '1'.
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      cin >> mat[i][j]; // @brief Read matrix element.
      if (mat[i][j] == 1) { // @brief Check if the current element is '1'.
        l = i; // @brief Store the row index of '1'.
        k = j; // @brief Store the column index of '1'.
      }
    }
  
  // @brief Calculate the minimum moves. The center is (2,2).
  // The number of moves is the sum of absolute differences in row and column indices (Manhattan distance).
  int count = 0;
  count = abs(l - 2); // @brief Calculate vertical moves.
  count = count + abs(k - 2); // @brief Add horizontal moves.
  
  cout << count; // @brief Print the total minimum moves.
  return 0;
}

