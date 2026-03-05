/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A, "Twins".
 * @details The program finds the minimum number of coins one needs to take from a
 * set so that the sum of the taken coins is strictly greater than the sum of the
 * remaining coins.
 *
 * Algorithm:
 * A greedy approach is used. To get the largest sum with the minimum number of
 * coins, the largest available coins should be taken first.
 * 1. Read all coin values and calculate their total sum.
 * 2. Define a target sum `k` which is half of the total sum. The goal is to
 *    collect a sum strictly greater than `k`.
 * 3. Sort the coins in ascending order.
 * 4. Iterate backward from the end of the sorted array (i.e., from the largest
 *    coin), accumulating the values and counting the number of coins taken.
 * 5. Stop as soon as the accumulated sum exceeds the target sum `k`.
 * 6. The number of coins taken is the answer.
 *
 * Note: The code uses a Variable Length Array (VLA) `a[n]`, which is a
 * non-standard C++ extension.
 *
 * Time Complexity: O(N log N) dominated by the sorting step.
 * Space Complexity: O(N) to store the coin values.
 */
#include <bits/stdc++.h>
using namespace std;
main() {
  int n;
  cin >> n;
  int a[n];
  int sum = 0;
  // Read all coin values and calculate the total sum.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }

  // The target is to get a sum strictly greater than half of the total.
  int k = sum / 2;
  int coin = 0;
  sum = 0; // The 'sum' variable is reused to store the sum of coins taken.
  int i = n - 1;

  // Sort coins in ascending order to easily access the largest ones from the end.
  sort(a, a + n);

  /**
   * @brief Greedily take the largest coins until the sum is sufficient.
   */
  while (1) {
    coin++;
    sum += a[i]; // Add the next largest coin.
    // Check if the sum of taken coins is strictly greater than half the total.
    if (sum > k) {
      cout << coin;
      break; // Exit once the condition is met.
    }
    i--; // Move to the next largest coin.
  }
}
