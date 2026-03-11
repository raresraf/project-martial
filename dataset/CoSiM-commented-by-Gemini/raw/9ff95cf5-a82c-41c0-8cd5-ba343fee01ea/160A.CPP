/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem (Codeforces 160A).
 * @details This program finds the minimum number of coins one needs to take from a
 *          set of coins so that the sum of values of the taken coins is strictly
 *          greater than the sum of values of the remaining coins.
 *
 * @algorithm The solution employs a greedy approach. To get the largest sum with the
 *            fewest coins, one must always pick the highest-value coins first.
 *            1. Read all coin values and calculate their total sum.
 *            2. Sort the coins in non-decreasing order.
 *            3. Calculate the target sum, which is half of the total sum. To have a sum
 *               strictly greater than the remainder, the sum of taken coins must exceed
 *               this half-sum.
 *            4. Iterate from the largest coin downwards, accumulating their sum and counting
 *               them, until the accumulated sum surpasses the half-sum threshold.
 *
 * @time_complexity O(N log N) dominated by the sorting operation.
 * @space_complexity O(N) to store the coin values in an array.
 */
#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int a[105]; // Array to store coin values, size 105 is a common competitive programming practice for N <= 100.
  int sm = 0; // Total sum of all coin values.

  // Read coin values and calculate the total sum.
  for (int i = 0; i < n; ++i) {
    cin >> a[i];
    sm += a[i];
  }

  // Sort coins in ascending order to easily access the largest ones.
  sort(a, a + n);

  float mid = sm / 2.0; // The threshold to pass: half of the total sum.
  int sum = 0;          // The sum of coins taken.
  int x = n - 1;        // Index to iterate from the largest coin.
  int coins = 0;        // Counter for the number of coins taken.

  /**
   * @brief Greedily take the largest coins.
   * The loop continues until the sum of taken coins is strictly greater
   * than half of the total sum.
   */
  while (sum <= mid && x >= 0) {
    sum += a[x];
    x--;
    coins++;
  }

  // Output the minimum number of coins required.
  cout << coins << "\n";
}
