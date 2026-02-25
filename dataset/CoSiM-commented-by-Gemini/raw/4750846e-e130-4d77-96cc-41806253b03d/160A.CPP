/**
 * @file 160A.CPP
 * @brief Solution for a greedy algorithm problem, likely from a programming contest ("Twins").
 * @details The problem is to find the minimum number of items (coins) to take from a set
 * such that the sum of the values of the taken items is strictly greater than the sum of the
 * values of the remaining items.
 *
 * Algorithm:
 * This solution uses a greedy strategy. To minimize the number of coins while maximizing
 * the sum, the highest-value coins should be taken first.
 * 1. Read all coin values.
 * 2. Sort the coins in descending (greatest to least) order.
 * 3. Iterate through the sorted coins, one by one, adding their value to a `taken_sum`
 *    and incrementing a `coin_count`.
 * 4. In each step, check if the `taken_sum` has become strictly greater than the sum
 *    of the remaining coins (`total_sum - taken_sum`).
 * 5. The loop terminates as soon as this condition is met, and the final `coin_count`
 *    is the minimum number required.
 *
 * Time Complexity: O(n log n), dominated by the sorting step.
 * Space Complexity: O(n) to store the coin values (using a VLA).
 */
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n, sum = 0;
  cin >> n;
  int i, a[n]; // Using a Variable Length Array (VLA), a non-standard C++ extension.
  
  // Read all coin values and calculate the total sum.
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  
  // Sort the coins in descending order to enable a greedy approach.
  sort(a, a + n, greater<int>());

  int coins = 0, taken = 0;
  
  // Manually process the first (largest) coin to initialize the state.
  taken = a[0];
  coins = 1;
  sum -= a[0]; // `sum` now represents the sum of remaining coins.
  i = 1;
  int j = n - 1; // `j` sets the boundary for the loop.

  /**
   * @block
   * @description This loop continues to take the next largest coins as long as
   * the sum of taken coins is not strictly greater than the sum of remaining coins.
   */
  while (i <= j) {
    // If the sum of taken coins already exceeds the rest, no more coins are needed.
    if (taken > sum)
      break;
    
    // Take the next largest coin.
    taken = taken + a[i];
    sum = sum - a[i];
    coins++;
    i++;
  }
  
  // Output the minimum number of coins needed.
  cout << coins;
}
