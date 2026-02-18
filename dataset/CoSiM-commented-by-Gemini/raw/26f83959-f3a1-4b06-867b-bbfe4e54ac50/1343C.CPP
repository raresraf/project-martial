/**
 * @file 1343C.CPP
 * @brief Solution to the "Alternating Subsequence" problem (1343C) from Codeforces.
 * @details The goal is to find an alternating subsequence (where signs of adjacent elements differ)
 * with the maximum possible sum. The key insight is to form this subsequence by picking the
 * largest number from each consecutive block of numbers with the same sign.
 *
 * @algorithm The solution uses a greedy approach to iterate through the array and identify
 * blocks of numbers with the same sign.
 * 1. For each test case, read the array `a`.
 * 2. Initialize `sum` to 0. Identify the sign of the first element.
 * 3. `mx` will track the maximum element within the current block of same-signed numbers.
 * 4. Iterate through the array starting from the second element:
 *    a. If the current element's sign is different from the previous block's sign:
 *       i. This marks the end of the previous block. Add the maximum element found in that block (`mx`) to the total `sum`.
 *       ii. Start a new block: update the sign and reset `mx` to the current element.
 *    b. If the current element's sign is the same:
 *       i. We are in the same block. Update `mx` if the current element is larger than the current `mx`.
 * 5. After the loop, add the maximum value from the very last block to the `sum`.
 * 6. Print the final `sum`.
 *
 * @complexity
 * Time Complexity: O(N) for each test case, as the algorithm involves a single pass through the array.
 * Space Complexity: O(N) to store the elements of the array.
 */
#include <bits/stdc++.h>
using namespace std;
// Macro to optimize I/O operations in competitive programming, common in Spanish-speaking communities.
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);

typedef long long int ll;

ll a[200010];

int main() {
  optimizar_io;
  int tc, n;

  cin >> tc;
  while (tc--) {
    cin >> n;
    for (int i = 0; i < n; i++)
      cin >> a[i];
    // Determine the sign of the first block. 1 for positive, 0 for negative.
    int prev = (a[0] > 0 ? 1 : 0), len = 1; // `len` appears unused.
    ll sum = 0, mx = a[0];                 // `mx` holds the max value in the current same-sign block.

    /**
     * @block
     * @brief Iterates through the array to find the max element in each alternating sign block.
     * @invariant `sum` holds the sum of maximums from completed blocks. `mx` holds the max
     *            of the current block being processed. `prev` stores the sign of the current block.
     */
    for (int i = 1; i < n; i++) {
      if (prev) { // 1 - Positivo: Currently in a block of positive numbers.
        if (a[i] < 0) { // Sign changes: the positive block ends.
          prev = 1 - prev;
          sum += mx; // Add the max of the completed positive block to the total sum.
          mx = a[i]; // Start a new negative block.
        } else {
          mx = max(mx, a[i]);
        }
      } else { // 0 - Negativo: Currently in a block of negative numbers.
        if (a[i] > 0) { // Sign changes: the negative block ends.
          prev = 1 - prev;
          sum += mx; // Add the max of the completed negative block to the total sum.
          mx = a[i]; // Start a new positive block.
        } else {
          mx = max(mx, a[i]);
        }
      }
    }
    // Add the maximum element from the very last block.
    sum += mx;
    cout << sum << "\n";
  }
  return 0;
}
