/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A - "Domino piling".
 * This program calculates the maximum number of 2x1 dominoes that can be placed on an M x N rectangular board.
 * Each domino covers two squares. The problem is a straightforward application of area division.
 * Algorithm: Direct calculation based on board area.
 * Time Complexity: O(1) - Involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries (e.g., iostream).

using namespace std; // Uses the standard namespace.

int main() {
  int n, m; // Declares integers to store the dimensions of the board (rows 'n', columns 'm').
  cin >> n >> m; // Reads the board dimensions from standard input.
  
  // Calculates the maximum number of dominoes.
  // The total area is n * m. Each domino covers 2 squares.
  // Integer division `(n * m) / 2` correctly handles cases where the total area is odd,
  // effectively leaving one square uncovered, which cannot fit another domino.
  cout << (n * m) / 2 << "\n";
  
  return 0; // Indicates successful program execution.
}
