/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 * @details The problem asks for the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * @algorithm
 * The solution is based on a simple area calculation.
 * 1. The total area of the board is M * N.
 * 2. Each domino covers an area of 2 units.
 * 3. The maximum number of dominoes is the total area divided by the area of
 *    a single domino.
 * 4. Using integer division, `(M * N) / 2` correctly computes `floor((M * N) / 2)`,
 *    which is the maximum number of non-overlapping dominoes that can fit.
 *
 * @complexity
 * Time Complexity: O(1), as the solution is a single arithmetic expression.
 * Space Complexity: O(1).
 */
#include <iostream>

int main() {
  int m, n; // Board dimensions M and N.
  std::cin >> m >> n;
  
  // The maximum number of 2x1 dominoes on an M*N board is floor((M*N)/2).
  // C++ integer division naturally performs the floor operation.
  std::cout << m * n / 2 << std::endl;
  
  return 0;
}
