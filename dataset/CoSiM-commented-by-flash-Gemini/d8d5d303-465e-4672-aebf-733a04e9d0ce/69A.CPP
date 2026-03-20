/**
 * @file 69A.CPP
 * @brief Determines if the sum of multiple 3D force vectors is zero.
 *        This implies that an object is in equilibrium (net force is zero).
 * Algorithm: Cumulative summation. Reads 'n' 3D vectors (x, y, z components)
 *            and sums their respective components into a single resultant vector.
 *            Finally, it checks if all components of the resultant vector are zero.
 * Time Complexity: O(N) where N is the number of force vectors, as each vector
 *                  is processed once.
 * Space Complexity: O(1) as only a few fixed-size integer variables and vectors are used.
 */
#include <bits/stdc++.h> // Includes common standard library headers, such as iostream and vector.

using namespace std;

/**
 * @brief Main function to compute the resultant force vector and check for equilibrium.
 * Functional Utility: Reads multiple 3D vectors and verifies if their sum equates
 *                     to a zero vector, indicating a state of equilibrium.
 */
int main() {
  int n; // Represents the number of force vectors to be read.
  cin >> n; // Reads the count of vectors from standard input.

  // 'a' is the resultant vector, initialized with three zero components.
  // 'b' is a zero vector, used for comparison.
  vector<int> a(3, 0), b(3, 0); 

  // Block Logic: Loops 'n' times, reading and accumulating each 3D force vector.
  // Invariant: After each iteration, vector 'a' holds the cumulative sum of
  //            all vectors processed so far.
  while (n--) {
    int q, w, e; // Variables to store the x, y, and z components of the current force vector.
    cin >> q >> w >> e; // Reads the three components for the current vector.

    // Block Logic: Accumulates the components of the current vector into the resultant vector 'a'.
    a[0] += q; // Sums x-components.
    a[1] += w; // Sums y-components.
    a[2] += e; // Sums z-components.
  }

  // Functional Utility: Compares the resultant vector 'a' with a zero vector 'b'.
  //                     If they are equal, it means the net force is zero.
  if (a == b) {
    cout << "YES"; // Prints "YES" if the object is in equilibrium.
  } else {
    cout << "NO";  // Prints "NO" if there is a net force.
  }

  return 0; // Indicates successful program termination.
}