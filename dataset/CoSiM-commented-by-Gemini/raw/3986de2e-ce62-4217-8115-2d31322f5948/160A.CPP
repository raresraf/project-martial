/**
 * @file 160A.CPP
 * @brief A buggy C++ solution for the "Twins" problem (Codeforces 160A).
 *
 * @details This program attempts to solve the "Twins" problem, which asks for the
 * minimum number of coins needed for one's share to be strictly greater than
 * the other's. The implementation uses a greedy strategy but contains several
 * significant bugs.
 *
 * @algorithm The intended algorithm is greedy:
 * 1. Read all coin values and compute their total sum.
 * 2. Sort the coins in ascending order.
 * 3. Iterate backward from the largest coin, accumulating a sum (`ans1`).
 * 4. Stop when `ans1` is greater than the remaining sum (`sum - ans1`).
 * 5. The number of coins taken should be the result.
 *
 * @note BUG: The variables `ans1`, `ans2`, and `ans` are not initialized,
 * leading to undefined behavior.
 * @note BUG: The final output is incorrect. The code prints the final index `i`
 * of the coin array, not the count of coins taken, which should be `n - i`.
 * @note Portability: The code uses a Variable-Length Array (VLA) `int a[n]`,
 * which is a non-standard C++ extension.
 *
 * @complexity
 * Time Complexity: O(N log N), dominated by the sorting operation.
 * Space Complexity: O(N), to store the coin values in the VLA.
 */
#include <algorithm>
#include <iostream>
#include <stdio.h> // stdio.h is included but not used.
using namespace std;
main() { // `main` should return `int`.
  int n, ans1, ans2, ans, i; // WARNING: ans1, ans2, ans are uninitialized.
  cin >> n;
  int a[n]; // Non-standard C++: Variable-Length Array (VLA).
  int sum = 0;
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }

  sort(a, a + n); // Sorts coins in ascending order.

  /**
   * @brief Block Logic: Greedily takes the largest coins until the sum is sufficient.
   * Iterates backward from the largest element in the sorted array.
   */
  for (i = n - 1; i >= 0; i--) {
    ans1 += a[i]; // WARNING: `ans1` is used uninitialized.
    ans2 = sum - ans1;

    if (ans1 > ans2) {
      ans = i; // `ans` now stores the *index*, not the count.
      break;
    }
  }
  // BUG: This prints the final index, not the number of coins taken (n - i).
  cout << ans << endl;
  return 0;
}
