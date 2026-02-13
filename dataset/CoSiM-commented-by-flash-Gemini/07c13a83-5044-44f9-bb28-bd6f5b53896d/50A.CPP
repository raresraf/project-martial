/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem. Calculates the maximum number of 1x2 dominoes
 * that can be placed on an M x N rectangular board.
 * Algorithm: The problem simplifies to dividing the total area of the board by the area of a single domino (2).
 * Time Complexity: O(1) due to direct arithmetic operations.
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <iostream>

using namespace std;

int main() {
  // Declares two integer variables to store the dimensions of the rectangular board.
  int m, n; 
  // Reads the two integer dimensions (M and N) from standard input.
  cin >> m >> n;
  // Block Logic: Calculates the maximum number of 1x2 dominoes that can be placed on an M x N board.
  // This is achieved by dividing the total area of the board (M * N) by the area of a single domino (2).
  // The integer division automatically handles the flooring, as partial dominoes cannot be placed.
  cout << (m * n) / 2;
  return 0;
}

