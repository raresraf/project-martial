#include <bits/stdc++.h>
using namespace std;

/**
 * @file 160A.CPP
 * @brief Solution to a competitive programming problem, "Twins".
 *
 * Problem Statement:
 * Given a set of coins, find the minimum number of coins to take such that
 * their sum is strictly greater than the sum of the remaining coins.
 *
 * Algorithm:
 * This solution uses a greedy approach. To take the minimum number of coins
 * while getting the largest possible sum, we should always take the coins with
 * the highest values first.
 *
 * 1. Read all coin values and calculate their total sum.
 * 2. Sort the coins in descending order (largest to smallest).
 * 3. Iterate through the sorted coins, accumulating a `total` and counting
 *    the coins (`cnt`). The loop continues as long as the accumulated `total`
 *    does not exceed half of the `sum`.
 * 4. The loop finds the maximum number of largest coins (`cnt`) whose sum is
 *    less than or equal to half the total sum.
 * 5. To have a sum *strictly greater* than the other half, we need to take
 *    one more coin. Therefore, the answer is `cnt + 1`.
 *
 * @note This code uses a Variable Length Array (VLA) `int arr[n]`, which is a
 * non-standard C++ extension supported by compilers like GCC but is not part
 * of the C++ standard. A `std::vector` would be a standard-compliant
 * alternative.
 *
 * Time Complexity: O(N log N) dominated by the sorting step.
 * Space Complexity: O(N) for storing the coin values.
 */

int main() {
  int n;
  cin >> n;

  // Using a non-standard Variable Length Array.
  int arr[n] = {0};

  int cnt = 0;   // Counts coins for the sum that is <= half of the total.
  int sum = 0;   // The total sum of all coins.
  int total = 0; // The sum of the largest coins being accumulated.

  // Block Logic: Read all coins and calculate the total sum.
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }

  // Sort the coins in descending order to process from largest to smallest.
  sort(arr, arr + n, greater<int>());

  // Block Logic: Find how many of the largest coins can be taken without
  // their sum exceeding half of the total sum.
  // Invariant: 'total' is the sum of the 'cnt' largest coins.
  for (int i = 0; i < n; i++) {
    // Pre-condition: Check if taking the current coin would keep the
    // accumulated sum at or below half of the total sum.
    if (total + arr[i] <= sum / 2) {
      cnt++;
      total += arr[i];
    } else {
      // If adding the next largest coin would exceed half the sum, we stop.
      break;
    }
  }

  // The number of coins required is `cnt` (the number of coins whose sum is <=
  // half) plus one more coin to guarantee the sum is strictly greater than half.
  cout << cnt + 1 << endl;
}