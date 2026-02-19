/**
 * @file 69A.CPP
 * @brief Solution to a competitive programming problem: "Young Physicist".
 *
 * This problem involves determining if a system of forces acting on an object is in equilibrium.
 * The object is in equilibrium if the sum of all force vectors is zero.
 * Given 'n' force vectors, each with x, y, and z components, the program calculates
 * the sum of components in each dimension and checks if all sums are zero.
 *
 * Algorithm: Vector sum calculation and equilibrium check.
 * Time Complexity: O(N), where N is the number of force vectors.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout
  // to optimize high-throughput I/O required for competitive programming constraints.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Variable Declarations:
  // i: Loop counter for iterating through force vectors.
  // n: Total number of force vectors.
  // x, y, z: Components of a single force vector.
  // sumx, sumy, sumz: Accumulators for the sum of components in each dimension.
  int i = 0, n, x, y, z, sumx = 0, sumy = 0, sumz = 0;

  // Input: Reads the total number of force vectors.
  cin >> n;

  // Block Logic: Iterates 'n' times to read each force vector and accumulate its components.
  // Invariant: At the start of each iteration, `i` represents the count of processed vectors.
  while (i < n) {
    // Input: Reads the x, y, and z components of the current force vector.
    cin >> x >> y >> z;
    // Functional Utility: Accumulates the x, y, and z components into their respective sums.
    sumx += x;
    sumy += y;
    sumz += z;
    i++; // Inline: Increments the counter for processed vectors.
  }

  // Block Logic: Checks if the system is in equilibrium.
  // Precondition: `sumx`, `sumy`, `sumz` hold the total sum of force components in each dimension.
  // An object is in equilibrium if the net force in all three dimensions is zero.
  if (sumx == 0 && sumy == 0 && sumz == 0)
    // Output: Prints "YES" if the object is in equilibrium.
    cout << "YES";
  else
    // Output: Prints "NO" if the object is not in equilibrium.
    cout << "NO";

  return 0; // Functional Utility: Indicates successful program execution.
}
