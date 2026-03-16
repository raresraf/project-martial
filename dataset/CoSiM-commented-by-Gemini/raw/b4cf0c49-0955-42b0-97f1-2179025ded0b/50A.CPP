/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 * @url https://codeforces.com/problemset/problem/50/A
 *
 * @details
 * The problem asks for the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N.
 *
 * @algorithm
 * The total area of the board is M * N. Each domino covers an area of 2.
 * Therefore, the maximum number of dominoes that can fit on the board is
 * the total area divided by the area of a single domino. Integer division
 * `(M * N) / 2` correctly gives the maximum number of full dominoes that can be placed.
 *
 * @complexity
 * Time Complexity: O(1), as it involves a single arithmetic calculation.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, m;
  cin >> n >> m;
  
  /**
   * @brief Calculate and print the maximum number of dominoes.
   * The formula `(n * m) / 2` computes how many 2x1 dominoes can fit
   * into an n x m area.
   */
  cout << (n * m) / 2 << "
";
  
  return 0;
}
