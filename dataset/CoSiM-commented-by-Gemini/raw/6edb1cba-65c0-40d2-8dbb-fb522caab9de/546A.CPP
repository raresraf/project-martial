
#include <bits/stdc++.h>
#include <string.h>
using namespace std;

/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A, "Soldier and Bananas".
 * @details Calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas with a linearly increasing cost.
 *
 * Algorithm:
 * The total cost of `w` bananas forms an arithmetic series: `k + 2k + ... + wk`.
 * This can be written as `k * (1 + 2 + ... + w)`. The program uses the direct
 * mathematical formula for the sum of the first `w` integers, `w*(w+1)/2`, to
 * calculate the total cost in a single step. It then compares this cost to the
 * soldier's initial money `n` to determine how much needs to be borrowed.
 *
 * Time Complexity: O(1), due to the direct formula.
 * Space Complexity: O(1).
 */

// --- Boilerplate and macros for competitive programming ---
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"
#define Max(x, y, z) max(x, max(y, z))
#define Min(x, y, z) min(x, min(y, z))
// Redefines `int` to `long long` to prevent integer overflow issues.
#define int long long
#define trace1(x) cerr << #x << ": " << x << endl
#define trace2(x, y) cerr << #x << ": " << x << " | " << #y << ": " << y << endl
#define trace3(x, y, z)                                                        \
  cerr << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << endl
// --- End of boilerplate ---

int32_t main() {
  IOS; // Fast I/O macro.
  
  // k: base cost of a banana
  // n: initial money
  // w: number of bananas to buy
  // ans: total cost
  int k, n, w, ans;
  cin >> k >> n >> w;

  // Calculate total cost using the formula for the sum of an arithmetic series.
  ans = k * w * (w + 1) / 2;

  /**
   * @brief Determine the amount to borrow.
   * If total cost `ans` is less than or equal to initial money `n`, the soldier
   * needs to borrow 0. Otherwise, the amount to borrow is `ans - n`.
   */
  if (ans < n) {
    cout << "0";
  } else
    cout << ans - n;

  return 0;
}
