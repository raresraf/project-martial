/**
 * @file raw/45a66302-cc6f-4ffb-9226-03ed4ac6c7db/160A.CPP
 * @brief Solution to Codeforces problem 160A - Twins.
 *
 * @algorithm
 * The problem asks for the minimum number of coins to take from a set such that
 * the sum of the taken coins is strictly greater than the sum of the remaining
 * coins.
 *
 * This is a classic greedy problem. To minimize the number of coins while
 * maximizing their sum, we should always pick the coins with the highest values
 * first.
 *
 * The algorithm is as follows:
 * 1. Read all the coin values and calculate their total sum (`total_sum`).
 * 2. The target is to accumulate a sum `my_sum` that is strictly greater than
 *    `total_sum / 2`.
 * 3. Sort the coin values in descending order.
 * 4. Iterate through the sorted coins, taking the largest ones first.
 * 5. In each iteration, add the coin's value to `my_sum` and increment a counter
 *    for the number of coins taken.
 * 6. Stop as soon as `my_sum` becomes greater than `total_sum / 2`.
 * 7. The value of the coin counter at this point is the minimum number required.
 *
 * The implementation sorts the array in ascending order and then iterates from
 * the end to the beginning to achieve the same effect.
 *
 * @complexity
 * - Time Complexity: O(N log N), dominated by the sorting of the coin array.
 * - Space Complexity: O(N), to store the `n` coin values in an array.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, i, sum = 0;
  
  // The array 'a' is declared and initialized but never used.
  int a[100];
  for (i = 1; i <= 100; i++)
    a[i] = 0;

  cin >> n;
  int arr[n];

  /**
   * @brief Read the coin values and calculate the total sum.
   */
  for (i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }

  // The goal is to get a sum strictly greater than half of the total.
  int div = sum / 2;

  // Sort the coins in ascending order.
  sort(arr, arr + n);

  int s = 0;     // The sum of coins taken.
  int coins = 0; // The number of coins taken.

  /**
   * @brief Greedily take the largest coins until our sum is greater than half.
   * We iterate from the end of the sorted array to get the largest coins first.
   */
  for (i = n - 1; i >= 0; i--) {
    // If our sum is already sufficient, we can stop.
    if (s > div)
      break;
    else {
      s += arr[i];
      coins++;
    }
  }

  cout << coins;
  return 0;
}
