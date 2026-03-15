/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem using a greedy approach.
 * @details The problem is to find the minimum number of coins to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * @algorithm
 * The optimal strategy is a greedy one:
 * 1. Read all coin values and calculate their total sum. The target is to collect a sum
 *    strictly greater than `total_sum / 2`.
 * 2. Sort the coins in non-decreasing order.
 * 3. Iterate from the largest coin downwards, accumulating their sum and counting how many
 *    coins are taken.
 * 4. Stop as soon as the accumulated sum exceeds the target (`total_sum / 2`).
 * 5. The number of coins taken at that point is the minimum required.
 *
 * @complexity
 * - Time: O(N log N), dominated by the sorting operation.
 * - Space: O(N), to store the values of the `n` coins in an array.
 *
 * @note The code uses a Variable Length Array (VLA) `long long a[n]`, which is a
 * non-standard C++ extension supported by some compilers like GCC.
 */
#include <bits/stdc++.h>

int main() {
  // n: The number of coins.
  long long n;
  // i: Loop counter.
  // sum: The total value of all coins.
  // ans: The accumulated value of the coins taken.
  // cnt: The number of coins taken.
  long long i, sum = 0, ans = 0, cnt = 0;
  
  std::cin >> n;
  // VLA to store the coin values.
  long long a[n];

  /**
   * @brief Input loop.
   * Reads each coin's value and calculates the total sum.
   */
  for (i = 0; i < n; i++) {
    std::cin >> a[i];
    sum += a[i];
  }

  // The target sum to surpass is half of the total sum.
  sum /= 2;
  
  // Sort the coins in non-decreasing (ascending) order.
  std::sort(a, a + n);

  /**
   * @brief Greedy selection loop.
   * Iterates backwards from the largest coin, adding its value to `ans`
   * and incrementing the coin count `cnt` until `ans` is strictly greater than `sum`.
   */
  for (i = n - 1; i >= 0; i--) {
    ans += a[i];
    cnt++;
    // Pre-condition: Check if the sum of the largest coins taken so far exceeds half the total sum.
    if (ans > sum) {
      // Post-condition: If it does, we have found the minimum number of coins, so we exit the loop.
      break;
    }
  }

  // Output the minimum number of coins required.
  std::cout << cnt;

  return 0;
}
