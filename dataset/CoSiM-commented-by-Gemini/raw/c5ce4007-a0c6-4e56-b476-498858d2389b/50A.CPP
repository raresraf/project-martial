/**
 * @file raw/c5ce4007-a0c6-4e56-b476-498858d2389b/50A.CPP
 * @brief Solves the classic domino tiling problem.
 * Given a rectangular board of size M x N, the program calculates the maximum
 * number of 2x1 dominoes that can be placed on the board.
 *
 * @algorithm
 * The problem is solved by a simple area calculation. The total area of the board
 * is M * N. Each domino has an area of 2. The maximum number of non-overlapping
 * dominoes is the total area of the board divided by the area of a single domino,
 * rounded down to the nearest integer.
 *
 * @time_complexity O(1), as it involves a single arithmetic calculation.
 * @space_complexity O(1), as only a fixed number of variables are used.
 */
#include <bits/stdc++.h> // Includes all standard C++ libraries, common in competitive programming.
using namespace std;

int main() {

  int m, n;
  cin >> m >> n; // Read the dimensions of the board.

  /**
   * @calculation
   * Calculates the maximum number of 2x1 dominoes. This is achieved by
   * finding the total area of the M x N board and performing an integer
   * division by 2, the area of a single domino.
   */
  cout << floor(n * m * 1 / 2) << endl;

  return 0;
}
