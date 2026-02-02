/**
 * @file 160A.CPP
 * @brief Solves a coin-splitting problem using a greedy algorithm.
 * @details The program determines the minimum number of coins a person needs to
 * take from a set so that the sum of their coins is strictly greater than the
 * sum of the remaining coins.
 *
 * @par Algorithm
 * This problem can be solved greedily. To minimize the number of coins taken
 * while maximizing their sum, one should always pick the coins with the largest
 * values first.
 * 1. Read all coin values and calculate the total sum.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins, accumulating their sum and counting
 *    how many coins are taken.
 * 4. Stop when the accumulated sum is greater than half of the total sum.
 * 5. The number of coins taken at this point is the minimum required.
 *
 * @par Time Complexity
 * O(N log N), dominated by the sorting operation, where N is the number of coins.
 *
 * @par Space Complexity
 * O(N) to store the coin values in an array.
 */
// Greedy - Sorting.

#include <algorithm>
#include <iostream>

using namespace std;

int main(void) {
  int n, totalSum = 0, count = 0, countSum = 0;
  cin >> n;
  // Note: Using a Variable Length Array (VLA), a non-standard C++ extension.
  int a[n];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    totalSum += a[i];
  }
  // Sort the coins in descending order to apply the greedy strategy.
  sort(a, a + n, greater<int>());
  
  /**
   * Block Logic: Greedily take the largest coins until the sum is greater
   * than the sum of the remaining coins.
   * Invariant: `countSum` is the sum of the `count` largest coins.
   */
  for (int i = 0; i < n; i++) {
    // The condition to stop is when our sum is greater than half the total.
    if (countSum <= totalSum / 2) {
      countSum += a[i];
      count++;
    } else {
      break;
    }
  }
  cout << count << endl;

  return 0;
}