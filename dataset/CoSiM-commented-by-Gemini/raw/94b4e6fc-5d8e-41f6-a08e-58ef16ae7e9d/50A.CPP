/**
 * @file 50A.CPP
 * @brief Calculates the maximum number of dominoes on a rectangular board.
 *
 * This program solves the "Domino piling" problem from Codeforces. It calculates
 * the maximum number of 2x1 dominoes that can fit on an M x N board. This
 * implementation explicitly makes the total area even before dividing by 2.
 *
 * @algorithm Mathematical Formula
 * @complexity
 * - Time: O(1), as it involves a fixed number of arithmetic operations.
 * - Space: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // M and N are the dimensions of the rectangular board.
  int M;
  int N;
  cin >> M;
  cin >> N;
  int area = M * N;

  /**
   * @brief Block Logic: Ensures the area is even before calculating the number of dominoes.
   * If the area is odd, one square will inevitably be left uncovered. This code
   * makes that explicit by decrementing the area.
   * Note: This `if` block is functionally redundant in C++ because integer division
   * `(M * N) / 2` already computes `floor((M * N) / 2)`, achieving the same result.
   */
  if (area % 2 != 0) {
    area--;
  }

  // The number of dominoes is the usable area divided by the area of one domino (2).
  cout << area / 2;
  return 0;
}
