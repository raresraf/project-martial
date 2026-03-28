/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces.
 *
 * This program determines the minimum number of coins a person needs to take
 * from a set so that the sum of their coins is strictly greater than the sum
 * of the remaining coins.
 *
 * @algorithm The solution uses a greedy approach.
 * 1. The total sum of all coin values is calculated.
 * 2. The coins are sorted in ascending order.
 * 3. The largest coins are taken one by one. In each step, the taken coin's
 *    value is added to a "my_sum" and subtracted from the "remaining_sum".
 * 4. This process continues until "my_sum" is strictly greater than
 *    "remaining_sum". The number of coins taken is the answer.
 *
 * @time_complexity O(N log N), dominated by the sorting operation.
 * @space_complexity O(N) to store the values of the coins.
 */
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, a, sum = 0, mine = 0, coins = 0;
  vector<int> v;

  cin >> n;

  // Read all coin values into a vector.
  for (int i = 0; i < n; ++i) {
    cin >> a;
    v.push_back(a);
  }

  // Calculate the total sum of all coins.
  for (auto x : v) {
    sum += x;
  }

  // Sort the coins in non-decreasing order to easily access the largest ones.
  sort(v.begin(), v.end());

  /**
   * @block_logic Greedily take the largest remaining coin until the sum of
   * taken coins (`mine`) is strictly greater than the sum of remaining coins (`sum`).
   */
  while (mine <= sum) {
    mine += v.back(); // Add the largest coin to my sum.
    sum -= v.back();  // Subtract it from the remaining sum.
    v.pop_back();     // Remove the coin from the set.
    coins++;          // Increment the count of taken coins.
  }

  cout << coins << endl;

  return 0;
}