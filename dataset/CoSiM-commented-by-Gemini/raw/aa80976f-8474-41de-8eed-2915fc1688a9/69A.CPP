/**
 * @file 69A.CPP
 * @brief Determines if a body is in equilibrium under the action of multiple forces.
 * This program checks if the vector sum of a given set of 3D force vectors is zero.
 * If the net force along all three axes (x, y, z) is zero, the system is in equilibrium.
 *
 * Algorithm:
 * 1. Read the total number of force vectors, 't'.
 * 2. Initialize three variables to store the sum of forces along the x, y, and z axes (s1, s2, s3) to zero.
 * 3. Loop 't' times:
 *    a. Read the x, y, and z components (a, b, c) of a force vector.
 *    b. Add the components to their respective sums (s1 += a, s2 += b, s3 += c).
 * 4. After the loop, check if all three sums (s1, s2, s3) are equal to zero.
 * 5. If s1, s2, and s3 are all zero, print "YES". Otherwise, print "NO".
 *
 * Time Complexity: O(T), where T is the number of force vectors, due to the single loop.
 * Space Complexity: O(1), as the memory usage is constant regardless of the number of vectors.
 */
#include <bits/stdc++.h> // Includes all standard C++ libraries; a common practice in competitive programming for convenience.
using namespace std;
// Type definitions for common data types to shorten code.
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;

int main() {
  int t; // Number of force vectors.
  int a, b, c; // Components of a single force vector (x, y, z).
  
  // Read the number of test cases (force vectors).
  cin >> t;
  
  // s1, s2, s3 will store the sum of forces along x, y, and z axes respectively.
  int s1 = 0, s2 = 0, s3 = 0;
  
  /**
   * Block Logic: Accumulate the components of all force vectors.
   * Invariant: Before each iteration, s1, s2, and s3 hold the net force components for the vectors processed so far.
   */
  while (t--) {
    cin >> a >> b >> c;
    s1 += a; // Summing x-components.
    s2 += b; // Summing y-components.
    s3 += c; // Summing z-components.
  }
  
  /**
   * Block Logic: Check for equilibrium condition.
   * A body is in equilibrium if the net force vector is the zero vector,
   * which means the sum of components along each axis must be zero.
   */
  if (s1 == 0 && s2 == 0 && s3 == 0)
    cout << "YES\n"; // System is in equilibrium.
  else
    cout << "NO\n"; // System is not in equilibrium.
}