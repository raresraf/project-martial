/**
 * @file 266A.CPP
 * @brief This program solves the "Stones on the Table" problem. It calculates
 *        the minimum number of stones to remove from a row such that no two
 *        adjacent stones have the same color.
 * Algorithm: Linear scan, Adjacency check. It iterates through the stones,
 *            counting how many times an adjacent pair has the same color.
 * Time Complexity: O(N) where N is the number of stones (length of the string),
 *                  due to a single linear pass.
 * Space Complexity: O(N) for storing the string of stone colors.
 */
/*
 ~ Author    : @tridib_2003
 ~ Title     : 266A. Stones on the Table
 ~ Link      : https://codeforces.com/contest/266/problem/A
*/

#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define PI 3.1415926535897932384626
#define MOD 1000000007
#define vi vector<int>
#define mk(arr, n, type) type *arr = new type[n];
#define FOR(a, c) for (int(a) = 0; (a) < (c); (a)++)
#define w(x)                                                                   \
  int x;                                                                       \
  cin >> x;                                                                    \
  while (x--)
/**
 * @brief Macro for Fast I/O.
 * Functional Utility: Disables synchronization between C++ standard streams and
 *                     C standard streams, and unties `cin` from `cout`,
 *                     leading to faster input and output operations.
 */
#define FIO                                                                    \
  ios::sync_with_stdio(false);                                                 \
  cin.tie(0);                                                                  \
  cout.tie(0);

int main() {
  FIO; // Calls the Fast I/O macro.

  int tc = 1; // Number of test cases (set to 1, as per comment below).
  // cin >> tc; // Commented out: option to read multiple test cases.
  /**
   * Block Logic: Main loop for processing test cases.
   * Invariant: Executes once for the single test case, or multiple times if `cin >> tc` was active.
   */
  while (tc--) {
    int n; // Number of stones.
    // Reads the number of stones.
    cin >> n;

    string s; // String representing the colors of the stones.
    // Reads the string of stone colors.
    cin >> s;

    int moves = 0; // Counter for the number of stones to remove.

    /**
     * Block Logic: Iterates through the string of stones, checking adjacent pairs.
     * Invariant: `moves` is incremented each time two adjacent stones have the same color.
     */
    for (int i = 0; i < n - 1; ++i) { // Loop up to n-1 to safely access s[i+1].
      if (s[i] == s[i + 1])           // Checks if current stone and next stone have same color.
        ++moves;                      // If they do, one stone needs to be removed.
    }
    // Prints the minimum number of moves (stones to remove).
    cout << moves << '\n';
  }

  return 0;
}