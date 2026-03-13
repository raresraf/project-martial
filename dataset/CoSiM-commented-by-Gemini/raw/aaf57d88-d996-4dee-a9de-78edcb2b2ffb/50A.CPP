/**
 * @file 50A.CPP
 * @brief Solves the "Domino Piling" problem.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * Algorithm:
 * 1. The total area of the board is M * N.
 * 2. Each domino has an area of 2x1 = 2.
 * 3. The maximum number of non-overlapping dominoes is the total area of the
 *    board divided by the area of a single domino.
 * 4. This is calculated using integer division: (M * N) / 2.
 *
 * Time Complexity: O(1), as it only involves a few arithmetic operations.
 * Space Complexity: O(1).
 */
#include <cstdio>

using namespace std;

int main() {
  int l, b; // Dimensions of the rectangular board (length and breadth).

  // Read the dimensions of the board.
  scanf("%d %d", &l, &b);

  // Calculate and print the maximum number of dominoes.
  // Integer division handles the case where the total area is odd.
  printf("%d\n", (l * b) / 2);
}