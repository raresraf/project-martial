/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem.
 *
 * The problem is to find the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N.
 *
 * @algorithm Area Calculation
 * @details The total area of the board is M * N. Each domino covers an area of 2.
 * The maximum number of non-overlapping dominoes that can fit on the board is
 * the total area of the board integer-divided by the area of a domino.
 *
 * @complexity
 * Time: O(1), as the solution involves a few simple arithmetic operations.
 * Space: O(1).
 */
#include <iostream>

// The <bits/stdc++.h> header is a non-standard GNU extension for convenience.
// #include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to handle input and calculate the result.
 */
int main(int argc, char const *argv[]) {
  // These lines disable synchronization with C stdio and untie cin/cout,
  // speeding up I/O operations in competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  
  // m, n: the dimensions of the rectangular board.
  int m, n;
  cin >> m >> n;
  
  // k: the total area of the board.
  int k = m * n;
  
  // Each domino covers 2 squares. The maximum number of dominoes is the
  // floor of the total area divided by 2. Integer division achieves this.
  int ans = k / 2;
  
  // Print the result.
  cout << ans;

  return 0;
}
