#include <algorithm>
#include <iostream>
using namespace std;

/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem from Codeforces (Problem 160A).
 * @details This program calculates the minimum number of coins one needs to take from a set
 * such that the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * Algorithm:
 * This problem can be solved with a greedy approach. To get the largest possible sum with the
 * fewest number of coins, one should always pick the coins with the highest values first.
 * 1. Read all coin values and calculate their total sum.
 * 2. The goal is to obtain a sum strictly greater than `total / 2`.
 * 3. Sort the coins in non-decreasing order.
 * 4. Iterate from the largest coin downwards, adding its value to a running `sum` and
 *    counting the number of coins taken.
 * 5. Stop as soon as the `sum` exceeds `total / 2`. The number of coins taken at this
 *    point is the minimum required.
 *
 * Time Complexity: O(N log N), dominated by the sorting of the coins.
 * Space Complexity: O(N) to store the values of the `n` coins.
 */
int main() {
  int n;
  cin >> n;

  int coins[n], total = 0, result = 0;
  // Read coin values and calculate the total sum.
  for (int i = 0; i < n; i++) {
    cin >> coins[i];
    total += coins[i];
  }

  // The target sum to exceed is half of the total.
  total /= 2;
  // Sort the coins to easily access the largest ones.
  sort(coins, coins + n);

  int sum = 0;
  /**
   * @brief Greedily take the largest coins.
   * This loop iterates from the end of the sorted array (largest values) backwards.
   * It accumulates the sum and counts the coins.
   */
  for (int i = n - 1; i >= 0; i--) {
    sum += coins[i];
    result++;
    // Check if the sum of taken coins is now greater than the sum of the rest.
    if (sum > total)
      break;
  }

  cout << result << endl;

  return 0;
}