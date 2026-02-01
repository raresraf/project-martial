/**
 * @file 50A.CPP
 * @brief Solves a classic competitive programming problem: "Domino piling".
 * This program calculates the maximum number of 1x2 dominoes that can be placed
 * on an M x N chessboard. Each domino covers exactly two squares.
 * Algorithm: Direct mathematical calculation. The total area of the board is M * N squares.
 * Since each domino covers 2 squares, the maximum number of dominoes that can be placed
 * is simply the total area divided by 2. Integer division correctly handles cases where
 * the total number of squares is odd (the remaining single square cannot be covered).
 * Time Complexity: O(1) - constant time, as it involves only a few arithmetic operations.
 * Space Complexity: O(1) - constant space, as it only stores a few integer variables.
 */
#include <iostream>

using namespace std;

main() {
  int n, m;
  // Functional Utility: Reads two integers, 'n' and 'm', representing the dimensions of the chessboard.
  cin >> n >> m;
  // Functional Utility: Calculates and prints the maximum number of dominoes that can be placed on an n x m board.
  // The formula (m * n / 2) uses integer division, which naturally handles odd total squares by truncating the result.
  cout << m * n / 2 << endl;
}
