/**
 * @file 69A.CPP
 * @brief Solution for CodeForces Problem 69A - "Young Physicist".
 *
 * @details
 * The problem asks to determine if a body is in equilibrium given a set of force vectors
 * applied to it. A body is in equilibrium if the vector sum of all forces is zero.
 * This means the sum of the x-components, the sum of the y-components, and the sum
 * of the z-components must all be zero.
 *
 * @algorithm
 * 1. Read the number of force vectors, `n`.
 * 2. Store all `n` force vectors (each with 3 components) in a 2D vector (`t`).
 * 3. Initialize three variables (`a`, `b`, `c`) to store the sum of x, y, and z components, respectively.
 * 4. Iterate through the 2D vector of forces:
 *    a. For each vector, add its first component to `a`, second to `b`, and third to `c`.
 * 5. After iterating through all vectors, check if `a`, `b`, and `c` are all zero.
 * 6. If they are, print "YES"; otherwise, print "NO".
 *
 * @time_complexity O(N), where N is the number of force vectors, due to the two passes
 *                  (one to read, one to sum).
 * @space_complexity O(N) to store all the force vectors in a 2D vector.
 */
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;
  
  // A temporary vector to read a single (x, y, z) force vector.
  vector<int> v(3);
  // A 2D vector to store all n force vectors.
  vector<vector<int>> t;

  /**
   * @brief Block to read all n vectors from input and store them.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> v[j];
    }
    t.push_back(v);
  }

  int a = 0, b = 0, c = 0; // Sum of x, y, and z components respectively.

  /**
   * @brief Block to sum the components of all stored vectors.
   * Invariant: `a`, `b`, and `c` hold the vector sum of all forces processed so far
   * in the outer loop.
   */
  for (int i = 0; i < t.size(); i++) {
    for (int j = 0; j < t[i].size(); j++) {
      if (j == 0)
        a += t[i][j]; // Accumulate x-components.
      else if (j == 1)
        b += t[i][j]; // Accumulate y-components.
      else
        c += t[i][j]; // Accumulate z-components.
    }
  }

  /**
   * @brief Check for equilibrium.
   * If the sum of all x, y, and z components is zero, the net force is zero.
   */
  if (a == 0 && b == 0 && c == 0)
    cout << "YES";
  else
    cout << "NO";
    
  return 0;
}
