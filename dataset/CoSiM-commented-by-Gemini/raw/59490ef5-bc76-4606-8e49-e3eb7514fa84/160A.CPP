/**
 * @file 160A.CPP
 * @brief An incomplete solution for the "Twins" problem from Codeforces (160A).
 * @details The problem asks for the minimum number of coins one needs to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the remaining ones.
 *
 * @algorithm
 * The intended algorithm for this problem is a greedy approach:
 * 1. Calculate the total sum of all coin values.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins (from largest to smallest), accumulating their sum
 *    and counting how many coins are taken.
 * 4. Stop when the accumulated sum is strictly greater than half of the total initial sum.
 *    The number of coins taken at this point is the minimum required.
 *
 * This implementation correctly reads the data, calculates the total sum, and sorts
 * the coins (in ascending order). However, the final crucial step of iterating
 * and accumulating the sum to find the minimum number of coins is missing from the
 * last loop.
 *
 * @complexity
 * Time Complexity: O(N log N) dominated by the sorting step.
 * Space Complexity: O(N) to store the coin values in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @variable n: The number of coins.
   * @variable sum: The total value of all coins.
   */
  int n, sum = 0;
  cin >> n;
  
  /**
   * @variable v: A vector to store the values of the individual coins.
   */
  vector<int> v;
  
  /**
   * @block
   * @brief Reads all `n` coin values, calculates their total sum, and stores them.
   */
  for (int i = 0; i < n; i++) {
    int tmp;
    cin >> tmp;
    sum += tmp;
    v.push_back(tmp);
  }

  // Sorts the coin values in ascending order.
  sort(v.begin(), v.end());

  /**
   * @block
   * @brief This loop is incomplete.
   * Functional Intent: This loop was likely intended to iterate backwards from the
   * end of the sorted vector (i.e., from the largest coin to the smallest) to
   * accumulate a sum and count the minimum number of coins needed to exceed half
   * the total sum. The core logic to solve the problem is missing here.
   */
  for (auto i = v.end(); i != v.begin(); --i) {
  }

  return 0;
}