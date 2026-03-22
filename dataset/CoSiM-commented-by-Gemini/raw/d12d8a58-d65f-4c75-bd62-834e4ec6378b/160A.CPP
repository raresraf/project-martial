/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy approach.
 *
 * The problem is to find the minimum number of coins to take from a set
 * so that their sum is strictly greater than the sum of the remaining coins.
 *
 * @b Algorithm: Greedy Strategy
 *   The optimal strategy is to pick the largest coins first. The code implements
 *   this by sorting the coins and then iterating from largest to smallest. In each
 *   step, it updates the sum of coins taken (`sum_com`) and the sum of coins
 *   remaining (`sum`), and stops as soon as the "taken" sum exceeds the "remaining" sum.
 *
 * @b Time_Complexity: O(N log N), dominated by the sorting step.
 * @b Space_Complexity: O(N) to store the coin values.
 *
 * @note The logic inside the loop is confusing due to the use of an
 * intermediate variable `sum1` that is reset in every iteration, but it is
 * functionally correct.
 */
// Twins
#include <bits/stdc++.h>

using namespace std;
int main() {
  // n: number of coins
  // i: loop counter
  // sum: tracks the sum of *remaining* coins (after initialization)
  // sum1: temporary variable to hold the value of the current coin
  // e: counts the number of coins taken
  // sum_com: accumulates the sum of *taken* coins
  int n, i, sum = 0, sum1 = 0, e = 0, sum_com = 0;
  cin >> n;
  int arr[n];
  
  // Read coins and calculate the initial total sum.
  for (i = 0; i < n; i++) {
    cin >> arr[i];
    sum = sum + arr[i];
  }
  
  // Sort coins in ascending order to access the largest ones from the end.
  sort(arr, arr + n);

  /**
   * @brief Greedily take the largest coins one by one.
   * This loop correctly updates the sum of taken coins (`sum_com`) and
   * the sum of remaining coins (`sum`) to check the break condition.
   */
  for (i = n - 1; i >= 0; i--) {
    sum1 = sum1 + arr[i];
    sum = sum - sum1;
    e++;
    sum_com = sum_com + sum1;
    sum1 = 0;
    if (sum_com > sum)
      break;
  }
  
  // Print the minimum number of coins required.
  cout << e;

  return 0;
}
