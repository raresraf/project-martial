/**
 * @file 160A.CPP
 * @brief Solution for the "Twins" problem (Codeforces 160A).
 * @author competitive-programmer
 *
 * @details The problem is to find the minimum number of coins to take from a set
 * such that the sum of the coins taken is strictly greater than the sum of the
 * remaining coins.
 *
 * Algorithm: Greedy Approach
 * 1. Read all the coin values.
 * 2. Calculate the total sum of all coins. The target is to collect a sum
 *    that is strictly greater than `total_sum / 2`.
 * 3. Sort the coins in descending order.
 * 4. Greedily take the largest coins one by one, adding their value to a running
 *    `coinsum` and counting how many coins are taken.
 * 5. Stop as soon as `coinsum` is greater than `total_sum / 2`.
 * 6. The number of coins taken is the minimum required.
 *
 * Time Complexity: O(N log N), dominated by the sorting operation.
 * Space Complexity: O(N) to store the coin values in a vector.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  std::vector<int> v(n);

  // Read all coin values into the vector.
  for (auto &i : v) {
    cin >> i;
  }

  // Sort the coins in descending order to apply the greedy strategy.
  sort(v.begin(), v.end(), greater<int>());
  
  // Calculate the total sum of all coins.
  size_t sum = accumulate(v.begin(), v.end(), 0);
  // The goal is to have a sum strictly greater than the sum of the rest,
  // which is equivalent to having a sum strictly greater than half of the total sum.
  sum /= 2;
  
  int count = 0;     // The number of coins taken.
  int coinsum = 0;   // The sum of the coins taken.
  
  /**
   * Block Logic: Greedily take the largest coins until their sum is greater
   * than the sum of the remaining coins.
   */
  for (int i = 0; i < n; ++i) {
    // Pre-condition for breaking: If our sum is already greater than half the
    // total, we have taken the minimum number of coins required.
    if (coinsum > sum) {
      break;
    } else {
      // Take the next largest coin.
      coinsum += v[i];
      count++;
    }
  }
  
  // Output the minimum number of coins required.
  cout << count << '\n';
  return 0;
}
