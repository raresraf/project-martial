/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem.
 *
 * This program calculates the maximum number of dominoes that can be placed
 * on a rectangular board of size M x N. Each domino covers exactly two squares.
 *
 * @algorithm
 * 1. Read the dimensions of the board, `m` and `n`.
 * 2. Calculate the total number of squares on the board, which is `m * n`.
 * 3. Since each domino occupies 2 squares, the maximum number of dominoes that
 *    can fit is the total area divided by 2.
 * 4. The calculation `(m * n) / 2` using integer division correctly gives this
 *    maximum number. If the total area is odd, one square will be left uncovered,
 *    and integer division naturally handles this by truncating the decimal.
 *
 * @complexity
 * Time Complexity: O(1), as it involves a single arithmetic calculation.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // `m` and `n` are the dimensions of the rectangular board.
  int m, n;
  cin >> m >> n;

  // The total area of the board is m * n.
  // Each domino covers 2 squares.
  // The maximum number of dominoes is the total area integer-divided by 2.
  cout << (m * n) / 2;
  
  // The return 0 is implicit at the end of main in modern C++.
}
