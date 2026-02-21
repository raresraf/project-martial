/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy algorithm.
 *
 * This program finds the minimum number of coins a person needs to take from a
 * collection so that the total value of their coins is strictly greater than
 * the total value of the remaining coins.
 *
 * Algorithm:
 * 1. The total sum of all coins is calculated and stored in `s1`.
 * 2. The coins are sorted in ascending order.
 * 3. The program iterates from the largest coin downwards. In each step:
 *    a. The current coin's value is added to `s2` (the sum of coins being taken).
 *    b. The same value is subtracted from `s1` (which now represents the sum of
 *       the remaining coins).
 *    c. A check is made to see if `s2` is strictly greater than `s1`.
 * 4. If `s2 > s1`, the condition is met. The number of coins taken (`n - i`) is
 *    the answer, which is printed, and the loop terminates.
 *
 * Time Complexity: O(N log N), where N is the number of coins, dominated by the
 * sorting operation.
 * Space Complexity: O(N) to store the coin values.
 *
 * @note The variables `s1` and `s2` are declared as global variables. While this
 * works, using local variables is generally better practice.
 */
#include <bits/stdc++.h>
using namespace std;

// Global variables to store the sum of remaining coins (s1) and taken coins (s2).
int s1, s2;

int main() {
  int n;
  cin >> n;
  int a[1000]; // Fixed-size array, assuming n <= 1000.
  
  // Read coins and calculate the total sum in s1.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    s1 += a[i];
  }

  // Sort the coins to easily access the largest ones.
  sort(a, a + n);
  
  /**
   * @brief Greedily take the largest coins until the sum is greater than the rest.
   * Pre-condition: `a` is sorted, `s1` holds the total sum.
   * Invariant: In each iteration, `s2` is the sum of the largest coins taken so far,
   * and `s1` is the sum of the remaining coins.
   */
  for (int i = n - 1; i >= 0; i--) {
    // Move the largest remaining coin from the "remaining" sum to the "taken" sum.
    s2 += a[i];
    s1 -= a[i];
    
    // Check if the sum of taken coins is now strictly greater.
    if (s2 > s1) {
      // If so, the number of coins taken is the answer.
      cout << n - i;
      break;
    }
  }
  return 0;
}