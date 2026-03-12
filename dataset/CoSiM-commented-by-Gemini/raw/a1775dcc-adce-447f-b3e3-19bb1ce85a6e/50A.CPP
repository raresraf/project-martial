/*
Codeforces Beta Round #47 - A. Domino piling
https://codeforces.com/problemset/problem/50/A
*/

/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem (Codeforces 50A).
 * @details The problem asks for the maximum number of 2x1 dominoes that can be
 *          placed on a rectangular board of size M x N.
 *
 * @algorithm The solution is based on a simple area calculation. The total area of
 *            the board is M * N. Each domino covers an area of 2. Therefore, the
 *            maximum number of non-overlapping dominoes that can fit is the total
 *            board area divided by the area of a single domino. Integer division
 *            ` (M * N) / 2 ` correctly gives this maximum number.
 *
 * @time_complexity O(1), as the solution involves a few arithmetic operations.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;
#define FAST_INP                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL)

int main() {
  // FAST_INP is a macro for standard I/O optimization in competitive programming.
  FAST_INP;
  int m, n;
  cin >> m >> n;
  // The maximum number of dominoes is the total area of the board divided by 2.
  cout << (m * n) / 2 << "\n";
  return 0;
}
