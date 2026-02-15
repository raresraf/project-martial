/**
 * @file 69A.CPP
 * @brief A flawed solution to the "Young Physicist" problem.
 * @details The problem is to determine if a body is in equilibrium by checking if the
 * vector sum of all forces is the zero vector. This implementation attempts a
 * two-pass algorithm but contains a major bug in the summation loop.
 *
 * @section algorithm Algorithm (Intended)
 * The intended algorithm is a two-pass process:
 * 1.  **Data Storage:** Read all `3 * n` vector components into a single array, using 1-based indexing.
 * 2.  **Summation:** Iterate through the stored components, summing the x, y, and z
 *     components into three separate variables (`sum1`, `sum2`, `sum3`).
 * 3.  **Equilibrium Check:** Check if all three sums are zero and print "YES" or "NO" accordingly.
 *
 * @section bug Bug Analysis
 * The second `for` loop, which is intended for summation, is critically flawed.
 * It iterates `3 * n` times instead of the correct `n` times. In each iteration, it
 * advances the indices `j`, `k`, and `l` by 3. This causes the loop to repeatedly
 * re-process the initial component indices and then quickly read far beyond the
 * bounds of the `arr` array, leading to undefined behavior and an incorrect result.
 *
 * @section complexity Complexity Analysis (Intended)
 * - Time Complexity: O(N), where N is the number of vectors.
 * - Space Complexity: O(N) to store all 3*N components in an array.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, arr[10000];
  // j, k, l are intended as indices for x, y, and z components respectively.
  int j = 1, k = 2, l = 3;
  // sum1, sum2, sum3 are for accumulating the sums of the components.
  int sum1 = 0, sum2 = 0, sum3 = 0;
  cin >> n;

  // Phase 1: Read all 3*n vector components into the array 'arr'.
  // Note: This code uses 1-based indexing for the array.
  for (int i = 1; i <= n * 3; i++) {
    cin >> arr[i];
  }

  // Phase 2: Attempt to sum the components.
  // BUG: This loop should run 'n' times, not '3*n' times.
  // This causes the indices j, k, and l to go out of bounds rapidly.
  for (int i = 1; i <= n * 3; i++) {
    // Intended to sum all x-components, but does it incorrectly.
    sum1 += arr[j];
    j = j + 3;
    // Intended to sum all y-components.
    sum2 += arr[k];
    k = k + 3;
    // Intended to sum all z-components.
    sum3 += arr[l];
    l = l + 3;
  }

  // Check if the sums are zero. The result of this check is unreliable due to the bug.
  if (sum1 == 0 && sum2 == 0 && sum3 == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
