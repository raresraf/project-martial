/**
 * @file 160A.CPP
 * @brief Solution for a competitive programming problem that involves selecting the minimum number of coins
 * to achieve a sum greater than the sum of the remaining coins.
 *
 * @details
 * Algorithm: Greedy Approach
 * 1. Read all coin values and calculate their total sum.
 * 2. Sort the coins in ascending order.
 * 3. Iterate through the sorted coins in descending order, accumulating their sum and counting the number of coins taken.
 * 4. Stop when the accumulated sum is strictly greater than half of the total sum.
 * 5. The count of coins taken is the minimal number required.
 *
 * Time Complexity: O(N log N) dominated by the sorting step, where N is the number of coins.
 * Space Complexity: O(N) to store the coin values in a vector.
 */
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  int n;
  vector<int> v;
  cin >> n;
  long sum = 0;
  /**
   * @brief Block to read coin values, calculate the total sum, and populate the vector.
   * Pre-condition: 'n' holds the number of coins to be read.
   * Invariant: 'sum' holds the sum of all coins read so far. 'v' contains the coins read.
   */
  for (int i = 0; i < n; i++) {
    int x;
    cin >> x;
    sum += x;
    v.push_back(x);
  }
  // Sort the coins in non-decreasing order to facilitate the greedy selection of the largest coins.
  sort(v.begin(), v.end());
  long sum2 = 0, count = 0;
  /**
   * @brief Greedily selects the largest coins until their sum is greater than the sum of the remaining coins.
   * Pre-condition: The vector 'v' is sorted in ascending order. 'sum' holds the total sum of all coins.
   * Invariant: 'sum2' holds the sum of the 'count' largest coins selected so far.
   */
  for (int i = n - 1; i >= 0; i--) {
    sum2 += v[i];
    count++;
    /**
     * @brief The condition checks if the sum of the selected coins ('sum2') is strictly greater than the
     * sum of the remaining coins. The sum of remaining coins is `sum - sum2`.
     * The check `sum2 > sum - sum2` is equivalent to `2 * sum2 > sum`.
     * For integer arithmetic, `sum2 >= sum / 2 + 1` correctly implements `sum2 > sum/2`.
     */
    if (sum2 >= sum / 2 + 1)
      break;
  }
  // Output the minimum number of coins required to have a sum greater than the rest.
  cout << count << "\n";
}