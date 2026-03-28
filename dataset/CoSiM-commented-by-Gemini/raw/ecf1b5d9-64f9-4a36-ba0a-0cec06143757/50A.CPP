/**
 * @file 50A.CPP
 * @brief Calculates the maximum number of dominoes on a rectangular board.
 *
 * This program solves the "Domino Piling" problem. Given a rectangular board of
 * size M x N, the goal is to find the maximum number of 2x1 dominoes that can
 * be placed on it.
 *
 * @b Algorithm: The solution is based on a simple mathematical formula. The total
 *    area of the board is M * N. Since each domino covers an area of 2, the
 *    maximum number of dominoes is the total area divided by 2, flooring the
 *    result.
 *
 * @b Time_Complexity: O(1), as the solution involves a constant number of
 *    arithmetic operations.
 * @b Space_Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int m, n;
  cin >> m >> n;

  /**
   * @brief The logic calculates the maximum number of dominoes.
   *
   * A board of size M x N has M*N squares. Each domino covers 2 squares.
   * Therefore, the maximum number of dominoes is floor((M * N) / 2).
   * In C++ integer arithmetic, `(m * n) / 2` correctly computes this value.
   * The if-else statement in the original code is redundant but functionally correct.
   */
  if (m % 2 == 0 || n % 2 == 0) {
    // If either dimension is even, the total area is even.
    cout << (m * n / 2);
  } else {
    // If both dimensions are odd, the total area is odd, leaving one square unused.
    cout << ((m * n) - 1) / 2;
  }

  return 0;
}
