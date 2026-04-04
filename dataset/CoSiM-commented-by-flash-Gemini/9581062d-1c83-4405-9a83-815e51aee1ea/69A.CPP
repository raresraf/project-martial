/**
 * @file 69A.CPP
 * @brief Determines if a system of forces acting on an object is in equilibrium.
 * * Algorithm: Vector Summation for Equilibrium.
 * Time Complexity: $O(N \times K)$ where N is the number of test cases and K is the number of forces per test case.
 * Space Complexity: $O(1)$ as only a constant number of variables are used to store sums.
 *
 * This program reads multiple sets of 3D force vectors and, for each set, calculates
 * the sum of their x, y, and z components. If all three component sums are zero,
 * the system is in equilibrium, and "YES" is printed; otherwise, "NO" is printed.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // Declaration of variables for force components and their sums.
  // x1, y1, z1: Components of a single force vector.
  // sum_1, sum_2, sum_3: Accumulators for the sum of x, y, and z components, respectively.
  // n: Number of force vectors in the current system.
  int x1, y1, z1, sum_1, sum_2, sum_3, n;
  
  // Block Logic: Outer loop to process multiple test cases (systems of forces).
  // Pre-condition: Standard input contains an integer 'n' indicating the number of forces.
  // Invariant: For each iteration, a complete system of 'n' forces is read and analyzed for equilibrium.
  while (cin >> n) {
    // Functional Utility: Initializes component sums to zero for the current system of forces.
    sum_1 = 0;
    sum_2 = 0;
    sum_3 = 0;
    
    // Block Logic: Inner loop to read 'n' force vectors and accumulate their components.
    // Pre-condition: 'n' specifies the count of force vectors to follow.
    // Invariant: After each iteration, one force vector's components are added to the respective sums.
    // Post-condition: `sum_1`, `sum_2`, `sum_3` hold the total sums of x, y, z components for the 'n' forces.
    while (n--) {
      cin >> x1 >> y1 >> z1;
      sum_1 += x1;
      sum_2 += y1;
      sum_3 += z1;
    }
    
    // Block Logic: Checks if the system of forces is in equilibrium.
    // Invariant: If all component sums are zero, the system is in equilibrium.
    // Post-condition: Prints "YES" if equilibrium, "NO" otherwise.
    if (sum_1 == 0 && sum_2 == 0 && sum_3 == 0)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
  return 0;
}
