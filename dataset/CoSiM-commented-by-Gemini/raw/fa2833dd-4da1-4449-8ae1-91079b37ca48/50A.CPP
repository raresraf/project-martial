/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * @algorithm
 * The total area of the board is M * N. Each domino covers an area of 2.
 * Therefore, the maximum number of non-overlapping dominoes that can fit on the board
 * is the total area divided by the area of a single domino. Integer division
 * naturally handles cases where the total area is odd, as the fractional part is discarded.
 *
 * @formula
 * Max Dominoes = floor((M * N) / 2)
 *
 * @time_complexity O(1) as the solution involves a single arithmetic calculation.
 * @space_complexity O(1) as only a few variables are used to store the dimensions.
 */
#include <iostream>

int main() {
  int m, n;

  // Read the dimensions of the rectangular board.
  std::cin >> m >> n;

  /**
   * Calculate and print the maximum number of dominoes.
   * The area of the board is m * n, and each domino covers 2 squares.
   * Integer division `(m * n) / 2` correctly computes the result.
   */
  std::cout << m * n / 2 << std::endl;

  return 0;
}
