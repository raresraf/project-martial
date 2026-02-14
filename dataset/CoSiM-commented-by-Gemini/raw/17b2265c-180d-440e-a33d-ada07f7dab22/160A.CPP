/**
 * @file 160A.CPP
 * @brief Solution to a problem about partitioning a set of coins.
 * @details The goal is to find the minimum number of coins needed to have a sum of value
 * greater than the sum of the remaining coins. This is a classic greedy algorithm problem.
 *
 * @section algorithm Algorithm
 * The problem can be solved using a greedy approach. To maximize our sum with the minimum number of coins,
 * we should always pick the coins with the largest values first.
 * 1. Read all the coin values and calculate the total sum of all coins.
 * 2. The target is to have a sum that is strictly greater than the sum of the remaining coins.
 *    This is equivalent to having a sum greater than `total_sum / 2`.
 * 3. Sort the coins in non-decreasing order.
 * 4. Iterate from the largest coin downwards, adding its value to a running sum (`my_sum`).
 *    Keep a count of how many coins have been taken.
 * 5. Stop when `my_sum` becomes greater than `total_sum / 2`.
 * 6. The number of coins taken up to that point is the minimum required.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n log n), dominated by the sorting step.
 * - Space Complexity: O(n) to store the coin values in an array.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

/**
 * @brief Main function to execute the coin selection logic.
 * @return 0 on successful execution.
 */
int main() {
  int n, total_sum = 0, my_sum = 0, count = 0;
  cin >> n;

  int a[n];
  // Read all coin values and calculate the total sum.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    total_sum += a[i];
  }

  // The target sum to beat is half of the total sum.
  int target_sum = total_sum / 2;
  
  // Sort the coins in ascending order to easily access the largest ones.
  sort(a, a + n);

  // Greedily take the largest coins until our sum is greater than the target.
  while (my_sum <= target_sum && count < n) {
    // Take the next largest coin from the end of the sorted array.
    my_sum += a[n - 1 - count];
    count++;
  }
  
  // Output the minimum number of coins required.
  cout << count;
  return 0;
}