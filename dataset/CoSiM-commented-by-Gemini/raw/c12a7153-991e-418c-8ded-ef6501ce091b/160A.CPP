/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy algorithm.
 * @details The goal is to find the minimum number of coins to take from a set
 *          such that the sum of the values of the chosen coins is strictly
 *          greater than the sum of the values of the remaining coins.
 *
 * Algorithm:
 *   The problem is solved using a greedy approach, which is optimal in this case.
 *   1. Calculate the total sum of all coin values.
 *   2. Sort the coins in descending order.
 *   3. Iterate through the sorted coins (from largest to smallest), adding their
 *      value to a running sum and counting the number of coins taken.
 *   4. Stop when the running sum is strictly greater than the sum of the
 *      remaining coins (total_sum - running_sum).
 *
 * Time Complexity: O(N log N), dominated by the sorting operation.
 * Space Complexity: O(N) to store the values of the N coins.
 */
#include <bits/stdc++.h>
using namespace std;

// Global variables to store state: total coin sum, number of coins taken,
// and the sum of coins taken.
int sum = 0, n, a[111111], k = 0, total = 0;

int main() {
  cin >> n;

  // Block Logic: Read all coin values and compute the total sum.
  for (int i = 0; i < n; i++)
    cin >> a[i], total += a[i];

  /**
   * Core Logic: Implements the greedy strategy by sorting the coins in
   * descending order. This ensures that we always consider the largest
   * available coin first, which is essential for minimizing the number of
   * coins taken.
   */
  sort(a, a + n, greater<int>());

  /**
   * Block Logic: Greedily take the largest coins one by one.
   * In each iteration, we add the coin's value to our `sum` and increment
   * the coin count `k`. The loop terminates as soon as our `sum` is
   * strictly greater than the sum of the coins left behind (`total - sum`).
   */
  for (int i = 0; i < n; i++)
    if (sum > total - sum)
      break;
    else
      sum += a[i], k++;

  // Output: Print the minimum number of coins required.
  cout << k;
}
