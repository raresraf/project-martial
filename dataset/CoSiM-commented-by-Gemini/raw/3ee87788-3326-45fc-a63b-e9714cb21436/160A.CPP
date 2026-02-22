/**
 * @file 160A.CPP
 * @brief Solution for the "Twins" problem (Codeforces 160A).
 *
 * @algorithm
 * The problem is to find the minimum number of coins to take from a set such that
 * the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * The optimal strategy is a greedy one:
 * 1. Read all coin values and calculate their total sum.
 * 2. Sort the coins in ascending order.
 * 3. Iterate backward through the sorted array (from largest to smallest coin).
 * 4. In each step, take the current coin, add it to a running sum (`sum`), and
 *    increment a coin counter.
 * 5. Continue this process as long as the sum of taken coins (`sum`) is not yet
 *    strictly greater than the sum of the remaining coins (`total_sum - sum`).
 * 6. The loop terminates when the condition is met, and the value of the coin
 *    counter at that point is the minimum number required.
 *
 * @complexity
 * - Time Complexity: O(N log N), dominated by the sorting operation.
 * - Space Complexity: O(N) to store the coin values in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int total_sum = 0;
  vector<int> m;
  /**
   * @brief Read all `n` coin values into a vector and calculate the total sum.
   */
  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;
    m.push_back(temp);
    total_sum += temp;
  }

  // Sort the coins in non-decreasing (ascending) order.
  sort(m.begin(), m.end());

  int sum = 0;   // The sum of coins taken by the twin.
  int coins = 0; // The number of coins taken.
  /**
   * @brief Greedily take the largest coins.
   * The loop iterates backward from the largest coin. It continues as long as
   * the twin's sum is not yet strictly greater than the remaining sum.
   */
  for (int i = n - 1; i >= 0 && sum <= total_sum - sum; i--) {
    // Take the next largest coin.
    sum += m[i];
    coins++;
  }

  cout << coins;
}
