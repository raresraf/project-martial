/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem using a two-pass approach.
 * @details The problem is to determine if a body is in equilibrium, which occurs if the
 * vector sum of all forces applied to it is the zero vector (0, 0, 0).
 * This implementation first stores all vector components in memory before summing them.
 *
 * @section algorithm Algorithm
 * The algorithm is implemented in two distinct phases:
 * 1.  **Data Storage:**
 *     a. Read the number of force vectors, `n`.
 *     b. Create a single `std::vector<int>` to store all vector components.
 *     c. Loop `n` times, reading the three components (x, y, z) of each vector and
 *        pushing them sequentially into the storage vector.
 * 2.  **Data Processing:**
 *     a. Initialize three sum variables `x`, `y`, and `z` to zero.
 *     b. Loop through the storage vector with a step of 3.
 *     c. In each step, add the elements at indices `i`, `i+1`, and `i+2` to the
 *        sums `x`, `y`, and `z`, respectively, accumulating the total force per axis.
 * 3.  **Equilibrium Check:**
 *     a. After the second loop, check if `x`, `y`, and `z` are all zero.
 *     b. If they are, print "YES"; otherwise, print "NO".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of vectors, as the code iterates through the input twice in separate, non-nested loops.
 * - Space Complexity: O(N), as it stores all 3*N vector components in a `std::vector`.
 *   (Note: A more space-optimal O(1) solution exists by summing components as they are read, without storing them).
 */
#include <bits/stdc++.h>
#include <vector>
using namespace std;
int main() {
  vector<int> vect; // A single vector to store all x, y, z components sequentially.
  int x, y, z, i, n;
  cin >> n;
  
  // Phase 1: Read all vector components and store them in the 'vect'.
  // The temporary variables x, y, z are used here just to read input.
  while (n--) {
    cin >> x;
    vect.push_back(x);
    cin >> y;
    vect.push_back(y);
    cin >> z;
    vect.push_back(z);
  }
  
  // Reset x, y, z to be used as sum accumulators for the three axes.
  x = y = z = 0;
  
  // Phase 2: Iterate through the stored components and calculate the total sum for each axis.
  for (i = 0; i < vect.size(); i += 3) {
    x = x + vect[i];      // Sum of x-components.
    y = y + vect[i + 1];  // Sum of y-components.
    z = z + vect[i + 2];  // Sum of z-components.
  }
  
  // Check if the net force vector is (0, 0, 0), which indicates equilibrium.
  if (x == 0 && y == 0 && z == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
