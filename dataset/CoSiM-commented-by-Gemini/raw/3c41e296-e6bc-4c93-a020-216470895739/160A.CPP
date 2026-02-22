/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem (Codeforces 160A).
 * 
 * @algorithm
 * The problem asks for the minimum number of coins to take such that the sum of
 * their values is strictly greater than the sum of the remaining coins. A greedy
 * approach is optimal here.
 * 1. Calculate the total sum of all coin values.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins, accumulating their sum and counting
 *    how many coins are taken.
 * 4. In each step, check if the accumulated sum is strictly greater than the
 *    sum of the remaining coins (total_sum - accumulated_sum).
 * 5. The first time this condition is met, the current count is the minimum
 *    number of coins required.
 *
 * @complexity
 * - Time Complexity: O(N log N) due to the sorting of the coin values.
 * - Space Complexity: O(N) to store the coin values in an array.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @brief Optimizes C++ standard streams for faster input/output operations.
   * This is a common practice in competitive programming to speed up I/O.
   */
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int n;
  cin >> n;
  // Using a Variable Length Array (VLA), which is a non-standard C++ extension.
  int arr[n];
  int sum = 0;     // Holds the total sum of all coin values.
  int money = 0;   // Accumulates the sum of coins taken.
  int countt = 0;  // Counts the number of coins taken.
  /**
   * @brief Reads all coin values and calculates their total sum.
   * Pre-condition: n is the number of coins to be read.
   * Invariant: After each iteration, `sum` contains the sum of coins read so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum = sum + arr[i];
  }
  // Sorts the coins in ascending order.
  sort(arr, arr + n);
  // Reverses the array to get coins in descending order, for the greedy approach.
  reverse(arr, arr + n);
  /**
   * @brief Greedily selects the largest coins until their sum is greater than
   * the sum of the remaining coins.
   * Pre-condition: The `arr` is sorted in descending order.
   * Invariant: `money` holds the sum of the `countt` largest coins. `sum` holds
   * the sum of the remaining n - `countt` coins.
   */
  for (int i = 0; i < n; i++) {
    // Take the next largest coin.
    money = money + arr[i];
    // Decrease the sum of remaining coins.
    sum = sum - arr[i];
    // Increment the count of coins taken.
    countt++;
    // Check if the sum of coins taken is strictly greater than the rest.
    if (money > sum) {
      // If so, we have found the minimum number of coins, so we can stop.
      break;
    }
  }
  cout << countt;
  return 0;
}
