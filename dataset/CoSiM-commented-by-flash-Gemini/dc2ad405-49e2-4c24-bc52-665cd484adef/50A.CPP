/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A - "Domino piling".
 * This program calculates the maximum number of 2x1 dominoes that can be placed on an M x N rectangular board.
 * Each domino covers two squares. The problem is a straightforward application of area division.
 * Algorithm: Direct calculation based on board area.
 * Time Complexity: O(1) - Involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) - Uses a constant amount of memory for variables.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * Functional Utility: Declares integer variables to store the board dimensions (M, N)
   * and the fixed size of a domino (2). 'result' will store the calculated maximum dominoes.
   */
  int M, N, domino = 2;
  int result;
  
  // Reads the board dimensions M and N from standard input.
  cin >> M >> N;
  
  /**
   * Block Logic: Calculates the maximum number of dominoes.
   * Logic: Divides the total area of the board (M * N) by the area covered by one domino (2).
   * Integer division inherently handles the case where the total area is odd,
   * effectively leaving one square uncovered, which cannot fit another domino.
   */
  result = (M * N) / domino;
  
  // Prints the calculated maximum number of dominoes to standard output.
  cout << result << endl;
  
  return 0;
}
