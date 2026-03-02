#include <bits/stdc++.h>

typedef long long ll;

using namespace std;

/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem from Codeforces (Problem 160A).
 * @details This program calculates the minimum number of coins one twin needs to take
 * so their total coin value is strictly greater than the other twin's total.
 *
 * Algorithm:
 * A greedy approach is used. To minimize the number of coins, the coins with the
 * highest values should be taken first.
 * 1. The program reads all coin values and calculates their total sum.
 * 2. The coins are sorted in ascending order.
 * 3. It then iterates from the largest coin downwards. In each step, it adds the
 *    coin's value to the twin's `current_value` and subtracts it from the `sum`
 *    (which now represents the value of the remaining coins).
 * 4. The loop continues until the `current_value` is strictly greater than the
 *    `sum` of the remaining coins. The number of coins taken is the answer.
 *
 * Time Complexity: O(N log N), where N is the number of coins, dominated by sorting.
 * Space Complexity: O(N) to store the coin values.
 */
int main() {
  int n, coins[100], sum = 0;

  cin >> n;

  // Read coin values and calculate the initial total sum.
  for (int i = 0; i < n; i++) {
    cin >> coins[i];
    sum += coins[i];
  }

  // Sort coins in ascending order to easily access the largest ones.
  sort(coins, coins + n);
  
  // `current_value` is the sum for the twin taking coins.
  // `ans` is the number of coins taken.
  int current_value = 0, ans = 0;

  /**
   * @brief Greedily take coins from largest to smallest.
   */
  for (int i = n - 1; i >= 0; i--) {
    /**
     * @brief Check if the current twin's sum exceeds the remaining sum.
     * This check evaluates the state from the *previous* iteration.
     */
    if (current_value > sum) {
      cout << ans << endl;
      return 0;
    }
    // Take the next largest coin.
    current_value += coins[i];
    // The `sum` variable now represents the sum of the *remaining* coins.
    sum -= coins[i];
    ans++;
  }
  // The final `cout` handles the case where all coins must be taken.
  cout << ans << endl;
  return 0;
}

// g++ -std=c++11 -O2 -Wall 160A.cpp -o out