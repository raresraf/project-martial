/**
 * @file 50A.CPP
 * @brief Solution for Codeforces problem 50A, "Domino piling".
 *
 * @description
 * This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * @algorithm
 * The problem can be solved with a simple mathematical formula.
 * 1. The total area of the board is M * N squares.
 * 2. Each domino covers 2 squares.
 * 3. The maximum number of dominoes is the total area of the board divided
 *    by the area of a single domino.
 * 4. Integer division `(M * N) / 2` correctly handles both even and odd
 *    total areas, yielding the maximum number of non-overlapping dominoes.
 *
 * @time_complexity O(1), as it's a direct calculation.
 * @space_complexity O(1).
 */
#include <iostream>
using namespace std;
int main() {
  // a and b are the dimensions of the rectangular board.
  int a, b;
  cin >> a >> b;

  // Calculate and print the maximum number of dominoes.
  // This is the total area of the board divided by the area of one domino (2).
  cout << (a * b) / 2;
  return 0;
}
