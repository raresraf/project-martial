/**
 * @file 50A.CPP
 * @brief This program solves the "Domino piling" problem (Codeforces 50A).
 *   It calculates the maximum number of 2x1 dominoes that can be placed on an M x N chessboard.
 *   This implementation uses a more detailed calculation considering different tiling strategies.
 *   Problem description: https://codeforces.com/problemset/problem/50/A
 * * Algorithm: Case-based tiling calculation.
 *   The core idea is that each 2x1 domino covers 2 squares. The total area of the M x N board is M * N.
 *   Thus, the maximum number of dominoes is `(M * N) / 2`. This solution uses a more elaborate
 *   approach involving considering odd/even dimensions and potentially different orientations.
 * Time Complexity: O(1) - constant time, as it involves a fixed number of arithmetic and bitwise operations.
 * Space Complexity: O(1) - constant space for a few integer variables.
 */
//============================================================================
// Name        : template.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;
// Functional Utility: Suppresses specific compilation warning messages, often for older C functions in newer compilers.
#define _CRT_SECURE_NO_DEPRECATE
// Functional Utility: Alias for `long long`, used for larger integer values.
typedef long long ll;
// Functional Utility: Alias for `std::vector<int>`.
typedef vector<int> vi;
// Functional Utility: Alias for `std::pair<int, int>`.
typedef pair<int, int> ii;
// Functional Utility: Alias for `std::vector<std::pair<int, int>>`.
typedef vector<ii> vii;
// Functional Utility: Alias for `std::set<int>`.
typedef set<int> si;
// Functional Utility: Alias for `std::map<std::string, int>`.
typedef map<string, int> msi;
// Functional Utility: Macros for accessing `first` and `second` elements of a pair.
#define x first
#define y second
// Functional Utility: Macro for `push_back` method of vectors.
#define pb push_back
// Functional Utility: Macro for `std::make_pair`.
#define mp make_pair
// Functional Utility: Macro for a common `for` loop pattern from `a` to `b` inclusive.
#define REP(i, a, b)                                                           \
  for (int i = int(a); i <= int(b); i++)
// Functional Utility: Macro for a common `for` loop pattern from `0` to `n-1`.
#define forn(i, n) for (int i = 0; i < (n); i++)
// Functional Utility: Macros for iterating through containers using iterators.
#define TRvi(c, it) for (vi::iterator it = (c).begin(); it != (c).end(); it++)
#define TRvii(c, it) for (vii::iterator it = (c).begin(); it != (c).end(); it++)
#define TRmsi(c, it) for (msi::iterator it = (c).begin(); it != (c).end(); it++)
// Constant: Represents a large integer value, often used for infinity in algorithms.
#define INF 2000000000 // 2 billion
// Functional Utility: Macros for initializing memory with specific patterns, useful for `memset`.
#define MEMSET_INF 127 // about 2B
#define MEMSET_HALF_INF 63 // about 1B
// Example usage comments for memset (part of the template, not used in this specific code).
// memset(dist, MEMSET_INF, sizeof dist); // useful to initialize shortest path
// distances memset(dp_memo, -1, sizeof dp_memo); // useful to initialize DP
// memoization table memset(arr, 0, sizeof arr); // useful to clear array of
// integers

/**
 * @brief Main function to calculate the maximum number of dominoes.
 * Functional Utility: Reads board dimensions, applies different tiling strategies,
 *                     and prints the maximum possible dominoes.
 */
int main() {
  // Functional Utility: Optimizes C++ standard I/O streams for faster execution.
  ios::sync_with_stdio(false);
  cin.tie(0);
  int m, n; // `m`, `n`: Dimensions of the chessboard.
  // Block Logic: Reads the two dimensions of the chessboard.
  cin >> m >> n;

  // Functional Utility: Assigns the larger dimension to `a` and smaller to `b` for canonical processing.
  int a = max(m, n);
  int b = min(m, n);
  int ans = 0; // Stores the final answer (maximum dominoes).

  // Conditional Logic: Handles edge cases where board dimensions are too small to place any dominoes.
  // If either dimension is less than 1, or if the larger dimension is less than 2, no dominoes can be placed.
  if (a < 2 || b < 1)
    ans = 0;
  else {
    int first = 0; // Stores the result of the first tiling strategy.
    /**
     * Block Logic: Calculates the maximum dominoes using the first tiling strategy.
     * This strategy appears to prioritize tiling along the `b` dimension.
     * Inline: `(b & 1)` checks if `b` is odd (returns 1 if odd, 0 if even) using bitwise AND.
     * If `b` is even, `a * (b / 2)` covers the entire board.
     * If `b` is odd, `a * (b / 2)` covers `b-1` columns, and `(b % 2) * (a / 2)` covers
     * the remaining single column (`a / 2` dominoes can be placed in a column of length `a`).
     */
    if ((b & 1)) { // If `b` is odd.
      first = (a * (b / 2)) + ((b % 2) * (a / 2));
    } else { // If `b` is even.
      first = (a * (b / 2));
    }

    int second = 0; // Stores the result of the second tiling strategy.
    /**
     * Block Logic: Calculates the maximum dominoes using the second tiling strategy.
     * This strategy appears to prioritize tiling along the `a` dimension.
     * Inline: `(a & 1)` checks if `a` is odd. Similar logic to `first` calculation, but roles of `a` and `b` are swapped.
     */
    if ((a & 1)) { // If `a` is odd.
      second = (b * (a / 2)) + ((a % 2) * (b / 2));
    } else { // If `a` is even.
      second = (b * (a / 2));
    }
    // Functional Utility: The final answer is the maximum of the two tiling strategies.
    // This handles cases where one orientation might allow more dominoes than the other (e.g., due to remaining odd dimensions).
    ans = max(second, first);
  }

  // Block Logic: Prints the calculated maximum number of dominoes followed by a newline.
  cout << ans << "\n";
  return 0;
}
