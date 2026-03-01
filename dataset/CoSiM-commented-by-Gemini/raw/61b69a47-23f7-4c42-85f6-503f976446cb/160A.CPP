/**
 * @file 160A.CPP
 * @brief Solution for the "Twins" problem.
 *
 * @problem Given a set of 'n' coins with different values, the task is to find the
 * minimum number of coins one needs to take so that the sum of their values is
 * strictly greater than the sum of the values of the remaining coins.
 *
 * @algorithm This is a classic greedy problem. To achieve the largest possible sum with
 * the fewest coins, we must always pick the highest-value coins first.
 * The algorithm is as follows:
 * 1. Read all coin values and calculate their total sum.
 * 2. Sort the coins in ascending order.
 * 3. Iterate through the sorted coins in reverse (from largest to smallest).
 * 4. In each step, take the current coin, add its value to a personal sum (`my_sum`),
 *    and subtract it from the `total_sum` (which now represents the sum of remaining coins).
 * 5. Increment a coin counter.
 * 6. After taking each coin, check if `my_sum` is strictly greater than `total_sum`.
 * 7. The first time this condition is met, the loop terminates, and the current
 *    coin count is the minimum required.
 *
 * @time_complexity O(N log N), dominated by the sorting step.
 * @space_complexity O(N) to store the coin values.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  // Note: Using a variable-length array (VLA) `a[n]`, which is not standard C++.
  int a[n];
  int total = 0; // Sum of all coins.
  int sum = 0;   // Sum of coins taken.

  // Read coin values and calculate the total sum.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    total += a[i];
  }
  // Sort coins in ascending order.
  sort(a, a + n);

  // Counter for the number of coins taken.
  int count = 0;

  // Greedily take the largest coins first by iterating backwards.
  for (int i = n - 1; i >= 0; i--) {
    // Take the current largest coin.
    sum += a[i];
    // The `total` now represents the sum of the remaining coins.
    total -= a[i];
    count++;
    
    // Check if the sum of taken coins is strictly greater than the remaining sum.
    if (sum > total)
      break; // If so, we have found the minimum number of coins.
  }

  cout << count << endl;

  return 0;
}
