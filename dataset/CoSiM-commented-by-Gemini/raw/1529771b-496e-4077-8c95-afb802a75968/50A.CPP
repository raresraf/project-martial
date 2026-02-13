/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem.
 * @details The program calculates the maximum number of 2x1 dominoes that can be placed on a
 * rectangular board of size M x N without overlapping.
 *
 * @section algorithm Algorithm
 * The solution is based on a simple area calculation.
 * 1. The total area of the rectangular board is `M * N`.
 * 2. Each domino has an area of `2 * 1 = 2`.
 * 3. To find the maximum number of non-overlapping dominoes that can fit, we can divide the
 *    total area of the board by the area of a single domino.
 * 4. Using integer division, `(M * N) / 2`, correctly gives this maximum number. If the total
 *    area is odd, the single remaining square cannot fit a whole domino, and integer
 *    division naturally discards this remainder.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(1), as the solution only involves reading two integers and performing a single arithmetic operation.
 * - Space Complexity: O(1), as only a few variables are used to store the dimensions.
 */
#include <iostream>

int main() {
  int m, n;
  // Read the dimensions of the board, M and N.
  std::cin >> m >> n;
  
  // The maximum number of dominoes is the total area of the board (m * n)
  // divided by the area of one domino (2). Integer division handles both
  // even and odd total areas correctly.
  std::cout << m * n / 2 << std::endl;
  
  return 0;
}