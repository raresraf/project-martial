/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem, checking for force equilibrium.
 * @details This program aims to determine if a body is in equilibrium by checking if the
 * vector sum of all forces acting upon it is zero. It stores all force components
 * in an array before attempting to sum them.
 *
 * @section algorithm Intended Algorithm
 * 1. Read the number of force vectors, `n`.
 * 2. Read all `3*n` force components into a single array `arr`. The components are
 *    stored sequentially (x1, y1, z1, x2, y2, z2, ...).
 * 3. Initialize three variables, `sum1`, `sum2`, and `sum3`, to 0 to hold the sum
 *    of forces along the x, y, and z axes, respectively.
 * 4. Iterate `n` times, summing the components for each axis. The x-components are at
 *    indices `j`, y-components at `k`, and z-components at `l`, where these
 *    indices are incremented by 3 in each step.
 * 5. After summing all components, check if `sum1`, `sum2`, and `sum3` are all zero.
 * 6. If they are, print "YES"; otherwise, print "NO".
 *
 * @section bug Implementation Bug
 * The second `for` loop, intended to sum the components, incorrectly iterates `n * 3`
 * times instead of `n` times. This causes the index variables `j`, `k`, and `l` to
 * be incremented too many times, leading to out-of-bounds array access and
 * undefined behavior. The code may appear to pass for some test cases due to
 * coincidence but is fundamentally incorrect.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), based on the intended logic. The flawed implementation
 *   is O(n*n) due to the nested effect of the loop structure, though it's primarily just incorrect.
 * - Space Complexity: O(n) to store the `3*n` force components in the array.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, arr[10001], j = 1, k = 2, l = 3, sum1 = 0, sum2 = 0, sum3 = 0;
  cin >> n;
  
  // Read all 3*n force components into the array, starting from index 1.
  for (int i = 1; i <= n * 3; i++) {
    cin >> arr[i];
  }
  
  // BUG: This loop should run 'n' times, not 'n * 3' times.
  // As written, it will access the array out of bounds.
  // The INTENT is to sum the x, y, and z components separately.
  for (int i = 1; i <= n; i++) { // Corrected loop would be `i <= n`
    sum1 += arr[j]; // Sum x-components
    j = j + 3;
    sum2 += arr[k]; // Sum y-components
    k = k + 3;
    sum3 += arr[l]; // Sum z-components
    l = l + 3;
  }
  
  // Check if the net force vector is (0, 0, 0).
  if (sum1 == 0 && sum2 == 0 && sum3 == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
