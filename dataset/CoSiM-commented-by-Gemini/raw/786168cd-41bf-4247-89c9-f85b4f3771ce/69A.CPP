/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem by checking for vector equilibrium.
 *
 * This program checks if a system of 3D forces is in equilibrium. It does so by
 * reading `N` vectors, summing their x, y, and z components, and verifying if
 * the total sum for each component is zero.
 *
 * @algorithm
 * 1. Read the number of vectors, `N`.
 * 2. Declare a 2D array `arr[N][3]` to store the components of the `N` vectors.
 *    Note: This uses a Variable Length Array (VLA), which is a non-standard C++
 *    extension (but commonly supported by compilers like GCC).
 * 3. Initialize sum variables `x`, `y`, and `z` to 0.
 * 4. Loop `N` times to process each vector.
 *    a. In each iteration, read the three components of a vector into the array
 *       (`arr[i][0]`, `arr[i][1]`, `arr[i][2]`).
 *    b. Immediately add these components to their respective running sums (`x`, `y`, `z`).
 * 5. After the loop, if the sums of all components are zero (`x == 0`, `y == 0`,
 *    and `z == 0`), the system is in equilibrium.
 * 6. Print "YES" for equilibrium, "NO" otherwise.
 *
 * @implementation_note
 * This implementation is efficient in its use of a single loop for both reading input
 * and calculating the sums. However, the `arr[N][3]` array is redundant, as the
 * component values are never used again after being added to the sums. A more
 * memory-efficient approach would read the components into temporary variables
 * within the loop instead of storing them in a large array.
 *
 * @complexity
 * Time Complexity: O(N), for the single loop over the N vectors.
 * Space Complexity: O(N) as written, due to the VLA `arr[N][3]`. An optimized
 * version would have O(1) space complexity.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int N;
  cin >> N;

  // Use a Variable Length Array (VLA) to store the vectors.
  // This is a non-standard C++ extension.
  // The array itself is redundant for this problem's logic.
  int arr[N][3];

  // Initialize sums for the x, y, and z components.
  int x = 0, y = 0, z = 0;

  /**
   * @brief Loop to read vectors and compute the sum of their components.
   */
  for (int i = 0; i < N; i++) {
    // Read the three components of the current vector.
    cin >> arr[i][0] >> arr[i][1] >> arr[i][2];
    
    // Add the components to their respective totals.
    x += arr[i][0];
    y += arr[i][1];
    z += arr[i][2];
  }

  // Check if the net force vector is the zero vector.
  if (x == 0 and y == 0 and z == 0)
    cout << "YES";
  else
    cout << "NO";

  // The `return 0;` is implicit at the end of `main`.
}
