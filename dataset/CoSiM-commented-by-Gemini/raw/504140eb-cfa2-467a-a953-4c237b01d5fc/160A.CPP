/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces.
 * @details The problem is to find the minimum number of coins to take from a set
 * so that the sum of the values of the taken coins is strictly greater than the
 * sum of the values of the remaining coins.
 *
 * @algorithm
 * This is a classic greedy algorithm problem. To minimize the number of coins
 * taken while maximizing their sum, one should always pick the coins with the
 * highest values first.
 * 1. Read all coin values and calculate their total sum.
 * 2. Sort the coins in non-decreasing order.
 * 3. Iterate from the largest coin downwards, accumulating the sum of the coins
 *    taken (`sum1`) and decrementing the total sum (`sum`).
 * 4. Keep a count of the coins taken.
 * 5. Stop when the sum of taken coins (`sum1`) becomes strictly greater than
 *    the sum of the remaining coins (`sum`).
 * 6. The number of coins taken up to that point is the answer.
 *
 * @complexity
 * Time Complexity: O(N log N), dominated by the sorting operation, where N is
 * the number of coins.
 * Space Complexity: O(N) to store the coin values in an array.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // sum: tracks the sum of remaining coins. Initialized with total sum.
  int sum = 0;
  // sum1: tracks the sum of coins taken by the player.
  int sum1 = 0;
  int n;
  cin >> n;
  int a[100]; // Array to store coin values. Assumes n <= 100.

  // Read coin values and calculate the total sum.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum = sum + a[i];
  }

  // Sort the coins in ascending order to easily access the largest ones.
  sort(a, a + n);
  
  int cnt = 0; // Counter for the minimum number of coins to take.

  /**
   * @block
   * @brief Greedily take the largest coins until our sum is greater.
   * @invariant In each iteration, we take the next largest available coin.
   * `sum1` is the sum of coins taken, `sum` is the sum of coins not yet taken.
   */
  for (int i = n - 1; i >= 0; i--) {
    // Condition check: As long as our sum is not strictly greater than the
    // remaining sum, we must take another coin.
    if (sum1 <= sum) {
      cnt++;
    } else {
      // Our sum is now greater, so we can stop.
      break;
    }
    // Move the current coin from the 'remaining' sum to the 'taken' sum.
    sum1 = sum1 + a[i];
    sum = sum - a[i];
  }
  
  cout << cnt;
}
