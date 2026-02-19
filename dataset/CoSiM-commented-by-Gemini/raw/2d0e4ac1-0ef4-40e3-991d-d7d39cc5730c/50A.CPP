/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem on Codeforces.
 * @details This program calculates the maximum number of dominoes that can be placed on a rectangular board of given dimensions.
 * The problem is available at: https://codeforces.com/problemset/problem/50/A
 *
 * Algorithm:
 * 1. The board has dimensions M x N. The total area is M * N squares.
 * 2. Each domino covers exactly 2 squares.
 * 3. To find the maximum number of dominoes that can be placed, we can cover the entire board.
 * 4. The total number of 2-square dominoes that can fit is the total area of the board divided by 2.
 * 5. The calculation `(M * N) / 2` using integer division gives the maximum number of non-overlapping dominoes.
 *
 * Time Complexity: O(1), as the solution involves a single arithmetic calculation.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */
// Codeforces
// 50A - Domino piling

#include <iostream>
using namespace std;

int main() {
  int n;
  int m;
  // Read the dimensions of the board.
  cin >> n >> m;

  // Calculate the maximum number of dominoes that can fit on the board.
  // The total area is n * m. Each domino covers 2 squares.
  // Integer division handles both even and odd total areas correctly.
  cout << n * m / 2;

  return 0;
}
