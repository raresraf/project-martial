/**
 * @file 50A.CPP
 * @brief Optimal solution for Codeforces Problem 50A, "Domino piling".
 * @details This program calculates the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N.
 *
 * Algorithm:
 * The total area of the board is M * N. Each domino covers an area of 2.
 * The maximum number of non-overlapping dominoes is the total area divided
 * by the area of a single domino, rounded down. In integer arithmetic, this
 * is simply `(M * N) / 2`.
 *
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int n, m;

  cin >> n >> m;

  /**
   * @brief Calculates and prints the maximum number of dominoes.
   * Integer division `(n * m) / 2` correctly computes `floor((n * m) / 2)`.
   */
  cout << (n * m) / 2;

  return 0;
}
