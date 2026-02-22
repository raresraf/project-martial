/**
 * @d63d81eb-e90a-4923-91cc-f6bb5450d081/50A.CPP
 * @brief Solution for a competitive programming problem involving tiling a chessboard with dominoes.
 *
 * This program calculates the maximum number of 1x2 dominoes that can be placed
 * on a given M x N chessboard. Since each domino covers exactly two squares,
 * the maximum number of dominoes is simply half the total number of squares,
 * rounded down if the total number of squares is odd.
 *
 * Algorithm:
 * 1. Read the two integer dimensions M and N of the chessboard.
 * 2. Calculate the total area of the chessboard: M * N.
 * 3. The maximum number of dominoes is (M * N) / 2 using integer division.
 *
 * Time Complexity: O(1) - constant time operations for input, calculation, and output.
 * Space Complexity: O(1) - uses a fixed amount of memory for variables.
 */

#include <bits/stdc++.h> // Functional Utility: Includes most standard C++ libraries, common in competitive programming.
#include <cstdio>        // Functional Utility: Specifically includes C-style input/output functions like scanf and printf.

using namespace std;

int main() {
  int n, m;
  // Functional Utility: Reads two integers 'n' (rows) and 'm' (columns) from standard input.
  // Pre-condition: Two integers representing chessboard dimensions are provided.
  scanf("%d %d", &n, &m);
  // Functional Utility: Calculates the maximum number of 1x2 dominoes that can fit on an n x m board.
  // The total number of squares is n * m. Each domino covers 2 squares.
  // Integer division automatically handles cases where n * m is odd, effectively rounding down.
  printf("%d\n", (n * m) / 2);
  return 0; // Functional Utility: Indicates successful program execution.
}
