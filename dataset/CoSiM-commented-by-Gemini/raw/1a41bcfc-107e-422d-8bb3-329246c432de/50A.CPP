/**
 * @file 50A.CPP
 * @brief Solution for the "Domino piling" problem.
 * @details This program calculates the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N.
 *
 * @section algorithm Algorithm
 * The solution is based on a simple area calculation.
 * 1. The total area of the rectangular board is `M * N`.
 * 2. Each domino has an area of `2 * 1 = 2` square units.
 * 3. The maximum number of dominoes that can fit on the board is the total
 *    area of the board divided by the area of a single domino.
 * 4. The formula is `floor((M * N) / 2)`. In C++, integer division `(m * n) / 2`
 *    naturally performs this floor operation.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(1), as the solution consists of a fixed number of
 *   arithmetic operations, regardless of the input size.
 * - Space Complexity: O(1), as it uses a constant amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to calculate the maximum number of dominoes.
 * @return 0 on successful execution.
 */
int main() {
  int m, n;
  
  // Read the dimensions of the rectangular board.
  cin >> m >> n;
  
  long long int ans; // Use long long to be safe, though int is sufficient for problem constraints.
  
  // The total area is m * n. Each domino covers 2 squares.
  // Integer division gives the maximum number of dominoes that can fit.
  ans = (m * n) / 2;
  
  cout << ans << endl;
  
  return 0;
}