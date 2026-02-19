/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy algorithm.
 * @details To find the minimum number of coins for a majority sum, this program
 * sorts the coins and greedily picks the largest ones until their sum
 * exceeds the sum of the remaining coins.
 *
 * @algorithm The algorithm works as follows:
 * 1. Read all `n` coin values and calculate their total sum.
 * 2. Sort the coins in ascending order.
 * 3. Iterate backwards from the largest coin, adding its value to a personal sum `s`
 *    and simultaneously subtracting it from the total sum `sum`.
 * 4. After each coin is taken, `s` represents the sum of coins taken, and `sum`
 *    now represents the sum of the coins not yet taken.
 * 5. The loop breaks when `s` becomes strictly greater than `sum`.
 * 6. The number of coins taken is calculated based on the loop's end index.
 *
 * @time_complexity O(N log N), dominated by the `std::sort` operation.
 * @space_complexity O(N) to store the coin values.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  // Using a variable-length array (VLA), a C99 feature supported as a g++ extension.
  int a[n];
  // `sum` will initially hold the total sum of all coins, then the sum of remaining coins.
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  sort(a, a + n);
  // `s` will hold the sum of coins taken by the twin.
  int s = 0;
  int i;
  /**
   * Block Logic: Greedily take the largest coins until their sum is a majority.
   * `s` accumulates the sum of taken coins, while `sum` is destructively modified
   * to represent the sum of the remaining coins.
   */
  for (i = n - 1; i >= 0; i--) {
    s += a[i];
    sum -= a[i];
    if (s > sum)
      break;
  }
  // The number of coins taken is the number of loop iterations, which is calculated as `n - i`.
  cout << n - i;
}