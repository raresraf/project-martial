/**
 * @file 160A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Twins").
 * The goal is to find the minimum number of coins to take from a set such that
 * the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * @algorithm Greedy Approach. The optimal strategy is to always take the coins
 * with the largest values first. The program sorts the coins and accumulates them
 * in descending order until their sum is greater than half of the total sum.
 *
 * @time_complexity O(N log N), dominated by the sorting step.
 * @space_complexity O(N) to store the coin values.
 */
#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n, arr[101];
  int res = 0; // Initially used to store the total sum of all coins.
  cin >> n;

  /**
   * @block
   * @brief Reads coin values and calculates their total sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    res += arr[i];
  }

  // Sorts the coins in ascending order.
  sort(arr, arr + n);

  // This loop appears to be for debugging and is not part of the core logic.
  for (int i = 0; i < n; i++) {
    cout << arr[i] << endl;
  }

  // The sum to beat is half of the total sum.
  int div = res / 2;
  int counter = 0; // Counts the number of coins taken.
  res = 0; // `res` is reused to store the sum of the coins taken.

  /**
   * @block
   * @brief Greedily takes the largest coins until their sum is sufficient.
   *
   * Invariant: `res` holds the sum of the `counter` largest coins.
   * The loop iterates from the largest coin downwards, accumulating their sum.
   * It stops as soon as the accumulated sum `res` is strictly greater than `div`
   * (half of the total sum).
   */
  for (int i = n - 1; i >= 0; i--) {
    res += arr[i];
    counter++;
    if (res > div)
      break;
  }
  cout << counter << endl;

  return 0;
}
