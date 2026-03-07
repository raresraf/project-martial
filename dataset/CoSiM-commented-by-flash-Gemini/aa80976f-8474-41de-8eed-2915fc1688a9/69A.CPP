/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" competitive programming problem (Codeforces 69A).
 * The problem states that an object is in equilibrium if the sum of all forces acting on it is zero.
 * The program reads the number of forces and then the x, y, and z components for each force.
 * It determines if the object is in equilibrium by checking if the sum of x-components,
 * sum of y-components, and sum of z-components are all independently equal to zero.
 * Algorithm: Iterative summation of force components with conditional equilibrium check.
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1), as only a few scalar variables are used.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

typedef long long ll;         // Defines 'll' as a shorthand for 'long long'.
typedef pair<int, int> pii;   // Defines 'pii' as a shorthand for 'pair<int, int>'.
typedef vector<int> vi;       // Defines 'vi' as a shorthand for 'vector<int>'.

int main() {
  // Declares 't' for the number of force vectors.
  // Declares 'a', 'b', 'c' for the components of each force vector (x, y, z respectively).
  // Declares 's1', 's2', 's3' to accumulate the sums of x, y, and z components.
  int t, a, b, c;
  // Reads the number of force vectors.
  cin >> t;
  int s1 = 0, s2 = 0, s3 = 0; // Initializes sum accumulators to zero.
  
  /**
   * Block Logic: Iterates 't' times (or until 't' becomes 0), once for each force vector.
   * Invariant: At the start of each iteration, 't' represents the remaining force vectors to process,
   * and 's1', 's2', 's3' hold the accumulated sums of components so far.
   * Pre-condition: 't' is a positive integer.
   * Post-condition: 's1', 's2', 's3' hold the total sums of x, y, and z components.
   */
  while (t--) {
    // Reads the x, y, and z components of the current force vector.
    cin >> a >> b >> c;
    // Accumulates the sums for each component.
    s1 += a;
    s2 += b;
    s3 += c;
  }
  
  /**
   * Conditional Logic: Checks if the object is in equilibrium.
   * An object is in equilibrium if and only if the sum of all force components
   * in each dimension (x, y, z) is zero.
   */
  if (s1 == 0 && s2 == 0 && s3 == 0)
    cout << "YES\n"; // Prints "YES" if in equilibrium.
  else
    cout << "NO\n";  // Prints "NO" if not in equilibrium.
  
  return 0; // Indicates successful program execution.
}