/**
 * @file 50A.CPP
 * @brief Solution for the "Domino piling" problem.
 * @details The problem asks for the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * @section algorithm Algorithm
 * The solution is based on a simple area calculation.
 * 1. The total area of the board is M * N.
 * 2. Each domino covers an area of 2 square units.
 * 3. Therefore, the maximum number of dominoes that can fit on the board is the
 *    total area of the board divided by the area of a single domino.
 * 4. The formula is `floor((M * N) / 2)`. Integer division in C++ naturally
 *    handles the floor operation.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(1), as the solution involves a fixed number of arithmetic operations.
 * - Space Complexity: O(1), as it uses a constant amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to calculate the maximum number of dominoes.
 * @return 0 on successful execution.
 */
int main() {
  int a, b;
  // Read the dimensions of the rectangular board.
  cin >> a >> b;
  
  // The total area of the board is a * b.
  // Each domino covers 2 squares.
  // The maximum number of dominoes is the total area divided by 2.
  cout << a * b / 2 << endl;
  
  return 0;
}
