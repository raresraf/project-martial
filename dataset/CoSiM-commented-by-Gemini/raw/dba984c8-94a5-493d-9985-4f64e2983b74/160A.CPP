/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A - "Twins".
 *
 * @details
 * This program solves the problem of finding the minimum number of coins to take
 * from a set, such that the sum of the taken coins is strictly greater than the
 * sum of the remaining coins.
 *
 * Algorithm:
 * A greedy approach is optimal here. To maximize the sum with the minimum number
 * of coins, we should always pick the highest-value coins first.
 * 1. Calculate the total sum of all coin values.
 * 2. Sort the coins in ascending order.
 * 3. Iterate through the coins from largest to smallest, accumulating their sum
 *    and counting the number of coins taken.
 * 4. After picking each coin, check if the accumulated sum is strictly greater
 *    than the sum of the remaining coins (total_sum - accumulated_sum).
 * 5. The first time this condition is met, the current count is the minimum
 *    number of coins required. Print the count and terminate.
 *
 * Time Complexity: O(N log N), dominated by the sorting step.
 * Space Complexity: O(N), to store the values of the N coins.
 */
#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  // a[n]: VLA to store coin values. s: total sum of all coins.
  int a[n], s = 0;

  /**
   * Block Logic: Reads all coin values and calculates their total sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    s += a[i];
  }

  // Sort the coins in non-decreasing order.
  sort(a, a + n);

  // c: current sum of coins taken. f: count of coins taken.
  int c = 0, f = 0;
  
  /**
   * Block Logic: Greedily take the largest coins until the sum condition is met.
   * The loop iterates backwards through the sorted array (from largest to smallest).
   */
  for (int i = n - 1; i >= 0; i--) {
    // Add the largest available coin to the current sum.
    c += a[i];
    // Increment the count of coins taken.
    f++;
    
    // Pre-condition: Check if the sum of taken coins is strictly greater than the
    // sum of the remaining coins.
    if (c > (s - c)) {
      cout << f << endl;
      break; // Found the minimum number of coins, so exit.
    }
  }
  return 0;
}
