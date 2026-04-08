/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces (Problem 160A).
 * @details The program finds the minimum number of coins one needs to take so that
 * the sum of their coins is strictly greater than the sum of the remaining coins.
 *
 * @algorithm
 * This solution uses a greedy approach with a mathematical simplification.
 * 1. Read all `n` coin values and calculate their total sum.
 * 2. The condition `my_sum > remaining_sum` is equivalent to
 *    `my_sum > total_sum - my_sum`, which simplifies to `2 * my_sum > total_sum`,
 *    or `my_sum > total_sum / 2`. The code uses this simplified target.
 * 3. It calculates `half_sum = total_sum / 2`.
 * 4. The coins are sorted in non-decreasing order.
 * 5. It iterates from the largest coin downwards, accumulating their sum and
 *    counting the number of coins taken.
 * 6. The process stops as soon as the accumulated sum is strictly greater than `half_sum`.
 * 7. The final count is the minimum number of coins required.
 *
 * @time_complexity O(N log N), dominated by the sorting step.
 * @space_complexity O(N) to store the coin values.
 */
#include <bits/stdc++.h>

int main() {
  long long n, i, sum = 0, ans = 0, cnt = 0;
  std::cin >> n;
  long long a[n];

  // Read coin values and calculate the total sum.
  for (i = 0; i < n; i++) {
    std::cin >> a[i];
    sum += a[i];
  }

  // Calculate the target sum we need to exceed (half of the total sum).
  sum /= 2;

  // Sort the coins to enable a greedy approach.
  std::sort(a, a + n);

  /**
   * @block_logic Greedily take the largest coins until their sum is greater than half the total sum.
   * @pre `a` is sorted, `sum` holds half the total value of all coins.
   * @post `cnt` will hold the minimum number of large coins needed to exceed `sum`.
   */
  for (i = n - 1; i >= 0; i--) {
    ans += a[i]; // Add the largest available coin.
    cnt++;       // Increment coin count.
    if (ans > sum) {
      break; // Stop once our sum is greater than the sum of the rest.
    }
  }

  std::cout << cnt;

  return 0;
}