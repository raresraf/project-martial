/**
 * @file 160A.CPP
 * @brief Solution for CodeForces Problem 160A - "Twins".
 * 
 * @details
 * The problem is to determine the minimum number of coins one needs to take from a set
 * so that the sum of the values of the chosen coins is strictly greater than the sum
 * of the values of the remaining coins.
 *
 * @algorithm Greedy Algorithm
 * 1. Calculate the total sum of all coin values.
 * 2. Sort the coins in non-decreasing order.
 * 3. Iterate from the largest coin downwards, accumulating their sum (`halfsum`).
 * 4. In each iteration, increment a counter (`cnt`) for the number of coins taken.
 * 5. Stop when the accumulated sum (`halfsum`) is strictly greater than half of the total sum.
 * 6. The value of `cnt` at this point is the minimum number of coins required.
 *
 * @time_complexity O(N log N), dominated by the sorting step.
 * @space_complexity O(N) to store the coin values in an array.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;         // Number of coins.
  int sum = 0;     // Total sum of all coin values.
  int halfsum = 0; // Accumulated sum of coins taken by the twin.
  int cnt = 0;     // Counter for the number of coins taken.
  
  cin >> n;
  int arr[n]; // Array to store the values of the coins.

  /**
   * @brief Read coin values and calculate the total sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }

  // Sort the coins in ascending order to easily access the largest ones.
  sort(arr, arr + n);

  /**
   * @brief Greedily take the largest coins until the sum is greater than the rest.
   * Invariant: `halfsum` contains the sum of the `cnt` largest coins.
   */
  for (int i = n - 1; i >= 0; i--) {
    halfsum += arr[i];
    cnt++;
    
    // Check if the sum of the taken coins is now strictly greater than the sum of the remaining coins.
    if (halfsum > sum / 2) {
      cout << cnt;
      break; // Found the minimum number of coins, so exit the loop.
    }
  }

  return 0;
}
