/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem.
 *
 * @problem A body is subjected to 'n' force vectors. Each force is given by its
 * x, y, and z components. The program must determine if the body is in
 * equilibrium.
 *
 * @algorithm The physical principle for equilibrium is that the net force on the body
 * must be zero. This means the vector sum of all forces is the zero vector.
 * In 3D space, this condition is met if and only if the sum of the x-components,
 * the sum of the y-components, and the sum of the z-components are all
 * individually equal to zero.
 * The algorithm sums up the components for each axis across all 'n' vectors
 * and then checks if all three sums are zero.
 *
 * @time_complexity O(N), where N is the number of force vectors, as the code
 * iterates through the vectors twice.
 * @space_complexity O(N) due to storing all vector components in arrays before processing.
 * Note: A more space-efficient O(1) solution is possible by summing the components
 * as they are read, without storing them.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;

  // Note: The use of variable-length arrays (VLAs) like x[n] is a C99 feature
  // and not standard C++, but it's supported as an extension by some compilers like g++.
  int x[n], y[n], z[n];
  // Variables to hold the sum of components for each axis.
  int sumx = 0, sumy = 0, sumz = 0;

  // Data Input Phase: Read all vector components into arrays.
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }

  // Component Summation Phase: Calculate the sum for each axis.
  for (int i = 0; i < n; i++) {
    sumx = sumx + x[i];
    sumy = sumy + y[i];
    sumz = sumz + z[i];
  }

  // Equilibrium Check: The body is in equilibrium if and only if the net force
  // along each axis is zero.
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
