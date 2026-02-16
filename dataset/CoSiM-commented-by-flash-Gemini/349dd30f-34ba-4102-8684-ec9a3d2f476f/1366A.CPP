/**
 * @file 1366A.CPP
 * @brief Solution for a competitive programming problem involving resource allocation or pairing.
 *
 * Algorithm: The problem involves optimizing the number of pairs that can be formed from two resources, `n` and `m`,
 *            under specific constraints. The solution iteratively forms pairs, prioritizing cases where one resource
 *            is significantly larger than the other (2*n <= m or 2*m <= n), or forming pairs optimally by taking
 *            half of the minimum of the two resources. This process continues until one or both resources are depleted.
 * Time Complexity: O(log(min(n, m))) in each test case, as `n` and `m` are reduced by a factor related to their minimum.
 *                  Overall O(T * log(min(n,m))) where T is the number of test cases.
 * Space Complexity: O(1) as it uses a few constant-size variables per test case.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main entry point of the program.
 *
 * Functional Utility: Processes multiple test cases. For each test case, it
 *                     reads two integers `n` and `m` (representing resources),
 *                     then calculates the maximum `count` of items that can be
 *                     produced or paired based on a specific set of rules.
 *                     The result `count` is printed for each test case.
 *
 * @return 0 on successful execution.
 */
int main() {

  int t; // @brief Number of test cases.
  cin >> t;
  // Block Logic: Iterate through each test case.
  // Invariant: `t` is decremented in each iteration until all test cases are processed.
  while (t--) {
    int n; // @brief First resource quantity.
    int m; // @brief Second resource quantity.
    cin >> n >> m;
    int count = 0; // @brief Accumulator for the total count of items produced/paired.

    // Block Logic: Continue processing as long as both resources `n` and `m` are available.
    // Invariant: In each iteration, `n` and/or `m` are reduced, ensuring termination.
    while (n > 0 && m > 0) {
      // Block Logic: Special case for highly imbalanced resources.
      // If one resource is at least double the other, optimize pairing.
      // Invariant: This condition handles cases where `min(n, m)` pairs are formed,
      //            and the smaller resource is completely consumed.
      if (2 * n <= m || 2 * m <= n) {
        count += min(m, n); // Functional Utility: Add the minimum of `m` or `n` to the total count.
        if (m < n) {
          m = 0; // Functional Utility: Deplete resource `m`.
        } else {
          n = 0; // Functional Utility: Deplete resource `n`.
        }
      } else {
        // Block Logic: General case for moderately balanced resources.
        // Invariant: `k` pairs are formed where `k` is half of the minimum resource,
        //            and resources are adjusted accordingly.
        int k = min(m, n) / 2; // @brief Number of pairs that can be formed based on half of the minimum resource.
        count += k; // Functional Utility: Add calculated pairs to the total count.
        if (m <= n) {
          n -= 2 * (m / 2); // Functional Utility: Adjust `n` after taking pairs from `m`.
          m -= m / 2; // Functional Utility: Adjust `m` after taking pairs.

        } else {
          m -= 2 * (n / 2); // Functional Utility: Adjust `m` after taking pairs from `n`.
          n -= n / 2; // Functional Utility: Adjust `n` after taking pairs.
        }
      }
    }
    // Functional Utility: Print the final accumulated count for the current test case.
    cout << count << endl;
  }
}