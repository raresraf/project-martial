/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem, checking for equilibrium.
 * @details This program determines if a body is in equilibrium by checking if the vector
 * sum of all forces acting upon it is zero. It does this by reading all force components
 * into a single vector and then summing the components for each axis.
 *
 * @section algorithm Algorithm
 * 1. Read the number of force vectors, `n`.
 * 2. Store all `3*n` force components (x, y, and z for each vector) sequentially in a single `std::vector`.
 * 3. Initialize three variables, `sum_x`, `sum_y`, and `sum_z`, to 0.
 * 4. Iterate through the vector of components with a step of 3.
 *    - In each step `i`, `vect[i]` is an x-component, `vect[i+1]` is a y-component, and `vect[i+2]` is a z-component.
 *    - Add each component to its corresponding sum variable.
 * 5. After iterating through all the components, check if `sum_x`, `sum_y`, and `sum_z` are all zero.
 * 6. If they are, print "YES" to indicate equilibrium. Otherwise, print "NO".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), as the program iterates through the `n` vectors (or `3*n` components) a constant number of times.
 * - Space Complexity: O(n), to store the `3*n` force components in the vector. This is less optimal than an O(1) space solution.
 */
#include <bits/stdc++.h>
#include <vector>
using namespace std;

/**
 * @brief Main function to check for force equilibrium.
 * @return 0 on successful execution.
 */
int main() {
  vector<int> vect;
  int x, y, z, i, n;
  
  cin >> n;
  
  // Read all 3*n components and store them in a single vector.
  for(int j=0; j<n; ++j) {
    cin >> x >> y >> z;
    vect.push_back(x);
    vect.push_back(y);
    vect.push_back(z);
  }
  
  // Reset x, y, z to be used as sum accumulators.
  x = y = z = 0;
  
  // Iterate through the vector with a stride of 3 to sum the components for each axis.
  for (i = 0; i < vect.size(); i += 3) {
    x += vect[i];     // Sum of x-components.
    y += vect[i + 1]; // Sum of y-components.
    z += vect[i + 2]; // Sum of z-components.
  }
  
  // Check if the net force vector is zero.
  if (x == 0 && y == 0 && z == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  
  return 0;
}
