/**
 * @file 160A.CPP
 * @brief Solution for Codeforces problem 160A, "Twins".
 *
 * @description
 * The problem asks for the minimum number of coins one needs to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the
 * remaining coins.
 *
 * @algorithm
 * A greedy approach provides the optimal solution. To reach the target sum with
 * the minimum number of coins, one should always pick the coins with the
 * largest values first.
 *
 * 1. Read all `n` coin values and calculate their total sum.
 * 2. The target is to have a sum strictly greater than `total_sum / 2`.
 * 3. Sort the coin values in ascending order.
 * 4. Iterate backward through the sorted array (from largest to smallest coin).
 * 5. In each iteration, take the current (largest) coin, add it to your sum,
 *    and increment a counter for the number of coins taken.
 * 6. Stop as soon as your sum exceeds half of the total sum.
 * 7. The value of the counter at that point is the minimum number of coins.
 *
 * This implementation uses a clever trick: it initializes a variable with
 * `total_sum / 2` and subtracts the taken coins from it, stopping when the
 * variable becomes negative.
 *
 * @time_complexity O(N log N), dominated by the sorting step.
 * @space_complexity O(N) to store the coin values.
 */
#include <algorithm>
#include <iostream>
using namespace std;
int a[111]; // Array to store coin values.
int main() {
  int n, sum = 0, cnt = 0;
  cin >> n;

  // Block Logic: Read coin values and calculate the total sum.
  for (int i = 0; i < n; ++i) {
    cin >> a[i];
    sum += a[i];
  }

  // Sort the coins in non-decreasing order.
  sort(a, a + n);

  // The target sum to exceed is half of the total. We use this as a countdown.
  sum /= 2;

  // Block Logic: Greedily take the largest coins until our share is strictly
  // greater than the other share.
  // The loop continues as long as our sum has not surpassed the target (`sum/2`).
  for (int i = n - 1; sum >= 0; --i) {
    // Subtract the largest remaining coin from the target sum.
    sum -= a[i];
    // Count this coin as taken.
    ++cnt;
  }
  cout << cnt;
}
