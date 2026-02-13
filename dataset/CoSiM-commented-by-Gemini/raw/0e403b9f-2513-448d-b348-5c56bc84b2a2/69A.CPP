/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem (Codeforces 69A).
 * @author competitive-programmer
 *
 * @details This program determines if a system of forces is in equilibrium.
 * It reads a set of 3D vectors representing forces and checks if their vector
 * sum is the zero vector. A system is in equilibrium if the sum of all forces
 * along each axis (x, y, z) is zero.
 *
 * Algorithm:
 * 1. Read the number of force vectors, n.
 * 2. Initialize three variables to store the sum of the x, y, and z components
 *    of all vectors to zero.
 * 3. Loop n times, reading the three components of each force vector.
 * 4. Add each component to its respective sum.
 * 5. After the loop, check if all three sums are zero.
 * 6. If all sums are zero, the system is in equilibrium ("YES"). Otherwise,
 *    it is not ("NO").
 *
 * Time Complexity: O(n), as we iterate through the n force vectors once.
 * Space Complexity: O(1), as we only use a few variables to store the sums.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  // n: the number of force vectors acting on the body.
  long long int n;
  cin >> n;
  
  // Accumulators for the sum of forces along the x, y, and z axes.
  long long int count1 = 0; // Represents the sum of x-components.
  long long int count2 = 0; // Represents the sum of y-components.
  long long int count3 = 0; // Represents the sum of z-components.

  /**
   * Block Logic: Read all 3*n components of the n vectors.
   * Logic: The loop iterates 3*n times to read all coordinates. The modulo
   * operator on the loop index `i` is used to differentiate between the
   * x, y, and z components of each vector.
   */
  for (int i = 0; i < 3 * n; i++) {
    long long int input;
    cin >> input;
    // Pre-condition: i % 3 == 0 corresponds to the x-component.
    if (i % 3 == 0)
      count1 += input;
    // Pre-condition: i % 3 == 1 corresponds to the y-component.
    else if (i % 3 == 1)
      count2 += input;
    // Pre-condition: i % 3 == 2 corresponds to the z-component.
    else
      count3 += input;
  }

  /**
   * Condition for Equilibrium: A body is in equilibrium if and only if the
   * net force acting on it is zero. This means the sum of the components
   * along each axis must be zero.
   */
  if (count1 == 0 && count2 == 0 && count3 == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
