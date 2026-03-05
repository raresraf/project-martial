/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A, "Twins".
 * @details The problem is to find the minimum number of coins to take from a set
 * such that the sum of the taken coins is strictly greater than the sum of the
 * remaining coins.
 *
 * Algorithm:
 * The problem is solved using a greedy approach. To achieve the largest possible
 * sum with the fewest coins, we should always pick the coins with the highest value first.
 * 1. Read all `n` coin values and calculate their total sum.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins, starting from the largest, adding their
 *    value to a running total for the coins being taken (`res`).
 * 4. Stop when the sum of the taken coins (`res`) is strictly greater than half
 *    of the total sum. The condition `res > sum - res` is equivalent to
 *    `2 * res > sum`, or `res > sum / 2`.
 * 5. The number of coins taken up to that point is the minimum required.
 *
 * Note: The code uses a Variable Length Array (VLA) `arr[n]`, which is a
 * non-standard C++ extension.
 *
 * Time Complexity: O(N log N) due to the sorting step.
 * Space Complexity: O(N) to store the coin values.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  int n, sum = 0;
  cin >> n;
  int res = 0, count = 0;
  int arr[n];

  /**
   * @brief Read all coin values and calculate the total sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }

  /**
   * @brief Sort the coins in descending order to enable the greedy approach.
   */
  sort(arr, arr + n, greater<int>());

  /**
   * @brief Greedily take the largest coins until their sum (`res`) is strictly
   * greater than the sum of the remaining coins (`sum - res`).
   * The condition `res <= sum / 2` is equivalent to `2 * res <= sum`. The
   * loop stops once `res` becomes `> sum / 2`.
   */
  while (res <= sum / 2) {
    res += arr[count];
    count++;
  }

  cout << count;
  return 0;
}
