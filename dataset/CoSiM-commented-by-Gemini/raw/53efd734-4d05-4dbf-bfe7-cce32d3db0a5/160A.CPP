/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces.
 * @details The goal is to find the minimum number of coins to take from a set
 * so that their sum is strictly greater than the sum of the remaining coins.
 *
 * @algorithm
 * The problem is solved using a greedy approach. To get the largest sum with
 * the fewest coins, we should always pick the highest-value coins first.
 * Let `total_sum` be the sum of all coins and `my_sum` be the sum of coins we
 * take. The condition is `my_sum > total_sum - my_sum`, which simplifies to
 * `2 * my_sum > total_sum`, or `my_sum > total_sum / 2`.
 *
 * 1. Read all coins and calculate their total sum (`tot`).
 * 2. Sort the coins in non-decreasing order.
 * 3. Iterate from the largest coin downwards. In each step:
 *    a. Add the coin's value to a running sum of taken coins (`temp`).
 *    b. Increment a counter for the number of coins taken (`ans`).
 *    c. Continue until the sum of taken coins (`temp`) is strictly greater
 *       than half of the total sum (`tot / 2`).
 * 4. The final count `ans` is the minimum number of coins required.
 *
 * @complexity
 * Time Complexity: O(N log N), dominated by the sorting step.
 * Space Complexity: O(N) to store the coin values.
 */
#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n, i;
  int ans(0);   // The result: number of coins taken.
  int temp(0);  // The sum of the coins taken.
  int tot(0);   // The total sum of all coins.
  
  cin >> n;
  
  // Using a Variable Length Array (VLA), a non-standard C++ extension.
  int coins[n];

  // Read all coin values and calculate the total sum.
  for (i = 0; i < n; i++) {
    cin >> coins[i];
    tot += coins[i];
  }
  
  // Sort the coins to easily access the largest ones.
  sort(coins, coins + n);

  /**
   * @block
   * @brief Greedily take the largest coins until their sum exceeds half the total.
   */
  for (i = n - 1; i >= 0 && temp <= tot / 2; i--) {
    temp += coins[i];
    ans++;
    
    // Note: This explicit break is redundant, as the main loop condition
    // `temp <= tot / 2` already handles termination correctly.
    if (temp > tot / 2)
      break;
  }
  
  cout << ans << endl;
  return 0;
}
