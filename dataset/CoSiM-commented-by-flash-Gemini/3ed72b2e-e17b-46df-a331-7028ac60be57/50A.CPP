/**
 * @file 50A.CPP
 * @brief Solution for competitive programming problem: "Domino piling".
 *
 * @details This program calculates the maximum number of 2x1 dominoes that can
 *          be placed on a rectangular board of size M x N without overlapping.
 *          Since each domino covers 2 squares, the maximum number of dominoes
 *          is half of the total area of the board (M * N / 2).
 *          Algorithm: Direct mathematical calculation.
 *          Time Complexity: O(1) as it involves a few constant-time arithmetic operations.
 *          Domain: Competitive Programming.
 */
#include <cstdio>
using namespace std;

int main() {
  // Declare integers 'M' and 'N' to store the dimensions of the board.
  int M = 0, N = 0;
  // Read the two dimensions (M and N) from standard input.
  // `scanf` is used for formatted input. `\n` in the format string might
  // consume trailing whitespace, which can be problematic in some competitive
  // programming setups if not handled carefully, but is often harmless here.
  scanf("%d %d\n", &M, &N);
  // Calculate the maximum number of dominoes and print the result.
  // The calculation M * N / 2 directly gives the maximum number of 2x1 dominoes
  // that can fit on an M x N board, as each domino covers 2 squares.
  printf("%d\n", M * N / 2);
  return 0; // Indicate successful program termination.
}
