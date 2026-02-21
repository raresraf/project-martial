/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem by finding the minimum number of coins
 *        to take for a sum greater than the remainder.
 *
 * This program implements a greedy approach to determine the minimum number of
 * coins one must take from a set such that the sum of the taken coins is
 * strictly greater than the sum of the remaining coins.
 *
 * Algorithm (Intended):
 * 1. Read the number of coins, `n`.
 * 2. Read all `n` coin values into an array `a` and compute their total `sum`.
 * 3. Sort the array of coins in ascending order.
 * 4. Iterate backward from the largest coin. In each step, add the coin's
 *    value to a running sum of taken coins (`ans1`).
 * 5. Check if `ans1` has become greater than the sum of the remaining coins (`sum - ans1`).
 * 6. If it has, the number of coins taken is the minimum required. The loop
 *    breaks, and the count of taken coins is printed.
 *
 * Time Complexity: O(N log N), dominated by the sorting operation.
 * Space Complexity: O(N) to store the coin values in an array.
 *
 * @note Bug Warning: The variables `ans1` and `ans` are used without being
 * initialized, leading to undefined behavior. The code likely works by chance
 * if the allocated memory for `ans1` happens to be 0. Furthermore, the program
 * prints the final index `i` (`ans = i`), not the count of elements taken,
 * which should be `n - i`.
 */
#include <algorithm>
#include <iostream>
#include <stdio.h> // stdio.h is included but not used.
using namespace std;
main() {
  int n, ans1, ans2, ans, i; // Warning: ans1 and ans are not initialized.
  cin >> n;
  int a[n]; // Non-standard C++: Variable-Length Array (VLA).
  int sum = 0;
  // Read all coin values and calculate the total sum.
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }

  // Sort the coins in ascending order.
  sort(a, a + n);

  /**
   * @brief Greedily take the largest coins and check the sum condition.
   * The loop iterates from the largest element downwards.
   * Pre-condition: `a` is sorted. `sum` is the total value of all coins.
   */
  for (i = n - 1; i >= 0; i--) {
    // Add the largest available coin to the sum of taken coins (`ans1`).
    ans1 += a[i];
    // `ans2` is the sum of the remaining coins.
    ans2 = sum - ans1;

    // Check if the sum of taken coins is strictly greater.
    if (ans1 > ans2) {
      // If the condition is met, store the index and break.
      ans = n - i; // The number of coins taken is n - i
      break;
    }
  }
  // The output is intended to be the count of coins, but the logic
  // as written has flaws. It would print `n-i` if `ans` was assigned that.
  // Instead it prints the final value of `ans`, which was calculated as `n-i`.
  cout << n - ans << endl; // Corrected logic would be to print `n-i` or `ans` directly.
                           // The current `n-ans` seems to be a mistake.
  return 0;
}
