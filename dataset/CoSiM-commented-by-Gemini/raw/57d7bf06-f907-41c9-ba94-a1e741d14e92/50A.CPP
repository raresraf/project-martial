/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 * @details The problem asks for the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N.
 *
 * @algorithm
 * The solution is based on a simple area calculation. The total area of the board is M * N.
 * Each domino covers an area of 2. Therefore, the maximum number of non-overlapping
 * dominoes that can fit on the board is the total area divided by the area of a single
 * domino. Integer division `(m * n) / 2` naturally computes `floor((M * N) / 2)`,
 * which gives the correct answer.
 *
 * @complexity
 * Time Complexity: O(1), as the solution involves a few arithmetic operations.
 * Space Complexity: O(1), as it only uses a few variables to store the dimensions.
 */
#include <iostream>
using namespace std;

int main() {
  /**
   * @variable m, n
   * @brief The dimensions (length and width) of the rectangular board.
   */
  int m, n;
  
  // Read the board dimensions from standard input.
  cin >> m >> n;

  /**
   * @brief Calculates and prints the maximum number of dominoes.
   * The total area of the board is `m * n`. Each domino covers 2 squares.
   * Integer division `(m * n) / 2` gives the maximum number of dominoes that can fit.
   */
  cout << int(m * n / 2) << endl;
}