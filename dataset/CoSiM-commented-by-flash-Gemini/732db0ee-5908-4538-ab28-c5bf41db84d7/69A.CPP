/**
 * @file 69A.CPP
 * @brief Solution to Codeforces Problem 69A: "Young Physicist".
 *
 * This problem states that a body is in equilibrium if the sum of all forces
 * acting on it is the zero vector. Given `n` force vectors, each with `x, y, z`
 * components, the task is to determine if the body is in equilibrium.
 *
 * Algorithm: Initialize a vector to accumulate the sums of the x, y, and z components
 * separately. Iterate through each force vector, adding its components to the
 * corresponding sums. After processing all forces, check if all three accumulated
 * sums are exactly zero.
 *
 * Time Complexity: O(N), where N is the number of force vectors. Each force vector
 *                  is processed in constant time.
 * Space Complexity: O(1) for storing the accumulator vector (fixed size of 3).
 *
 * Problem Link    :   https://codeforces.com/problemset/problem/69/A
 * Author          :   Isanchez_Aguilar
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, like iostream, vector, and algorithm.

using namespace std;

/**
 * @brief Main function to solve the "Young Physicist" problem.
 * Sets up fast I/O, reads the number of force vectors and their components,
 * calculates the resultant force vector, and determines if the body is in equilibrium.
 */
int main(void) {
  // Functional Utility: Optimizes C++ standard streams for faster input/output.
  ios_base::sync_with_stdio(0); // Disables synchronization with C stdio.
  cin.tie(0);                   // Unties cin from cout.

  int n; // Stores the number of force vectors.
  cin >> n; // Reads the total number of force vectors.

  // Vector to store the accumulated sums of x, y, and z components.
  // Initialized to size 3 with all elements set to 0 by default.
  vector<int> ans(3);

  /**
   * Block Logic: Iterates through each of the `n` force vectors.
   * Pre-condition: `n` is the total number of force vectors. `ans` is initialized to {0, 0, 0}.
   * Invariant: After each iteration, `ans` contains the cumulative sum of components
   *            for the forces processed so far.
   */
  for (int i = 0; i < n; ++i) {
    int x, y, z; // Temporary variables to read the components of the current force vector.
    cin >> x >> y >> z; // Reads the x, y, and z components.

    ans[0] += x; // Accumulate x-components.
    ans[1] += y; // Accumulate y-components.
    ans[2] += z; // Accumulate z-components.
  }

  /**
   * Block Logic: Determines if the body is in equilibrium.
   * Uses `std::count` to check if all three accumulated sums in the `ans` vector are zero.
   * If all three components of the resultant vector are zero, the body is in equilibrium.
   */
  if (count(begin(ans), end(ans), 0) == 3) {
    cout << "YES\n"; // Body is in equilibrium.
  } else {
    cout << "NO\n";  // Body is not in equilibrium.
  }

  return 0; // Indicates successful execution.
}