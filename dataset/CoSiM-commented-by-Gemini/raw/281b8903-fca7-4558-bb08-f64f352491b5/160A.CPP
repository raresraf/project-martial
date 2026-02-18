/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem (160A) from Codeforces.
 * @details The goal is to find the minimum number of coins to take such that the sum of their values
 * is strictly greater than the sum of the remaining coins.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This solution uses a greedy approach. It sorts the coins in descending order and
 * takes them one by one until the sum of taken coins exceeds the sum of the remaining coins.
 * 1. Read the number of coins (n) and their values into an array. Calculate the total sum.
 * 2. Sort the array of coins in descending order.
 * 3. Initialize `me` (sum of coins taken) to 0 and `ans` (count of coins taken) to 0.
 * 4. Iterate through the sorted coins:
 *    a. Check if the sum of coins taken (`me`) is already greater than the sum of remaining coins (`sum`). If so, break.
 *    b. Increment the count of coins taken (`ans`).
 *    c. Add the current coin's value to `me`.
 *    d. Subtract the current coin's value from `sum`.
 * 5. Print the final count `ans`.
 *
 * @complexity
 * Time Complexity: O(N log N), dominated by the sorting operation.
 * Space Complexity: O(N), for storing the coin values in an array.
 */
#include <bits/stdc++.h>
using namespace std;
// Type definitions and macros common in competitive programming templates.
typedef long long int ll;
typedef long double ld;
typedef pair<ll, ll> pll;
#define pb push_back
#define mp make_pair
#define f first
#define s second
#define gcd(a, b) __gcd(a, b)
#define lpr(i, s, e) for (ll i = s; i >= e; i--)
#define lpi(i, s, e) for (ll i = s; i <= e; i++)
#define lp(n) for (ll i = 0; i < n; i++)
int main() {
  // Fast I/O setup
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  int n, ans = 0, sum = 0, me = 0;
  cin >> n;
  // A variable-length array (VLA) is a non-standard C++ extension.
  int arr[n];
  /**
   * @block
   * @brief Reads coin values and calculates the total sum.
   */
  lp(n) {
    cin >> arr[i];
    sum += arr[i];
  }
  // Sort coins in descending order to greedily pick the largest ones first.
  sort(arr, arr + n, greater<int>());
  /**
   * @block
   * @brief Greedily takes the largest coins until the sum is sufficient.
   * @invariant `me` holds the sum of the `ans` largest coins. `sum` holds the sum of remaining coins.
   * @pre `arr` is sorted in descending order.
   * @post `ans` holds the minimum number of coins needed.
   */
  lp(n) {
    // If the sum I have (`me`) is already greater than what's left (`sum`), I'm done.
    if (me > sum)
      break;
    ans++;
    me += arr[i];
    sum -= arr[i];
  }
  cout << ans << endl;
  return 0;
}