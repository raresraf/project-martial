/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem by checking for vector equilibrium.
 *
 * This program determines if a system is in equilibrium by reading `n` 3D vectors
 * and checking if their vector sum is the zero vector. This is true if and only
 * if the sum of the x-components, y-components, and z-components are all individually zero.
 *
 * @algorithm
 * 1. Read the number of vectors, `n`.
 * 2. Declare arrays `x`, `y`, and `z` to store the vector components.
 *    (Note: These arrays are redundant in this implementation).
 * 3. Initialize three sum variables, `xsum`, `ysum`, and `zsum`, to 0.
 * 4. Loop `n` times to process each vector.
 *    a. Read the three components of the current vector into the arrays.
 *    b. Immediately add the component values to their respective sum variables.
 * 5. After the loop, all components have been summed up.
 * 6. Check if `xsum`, `ysum`, AND `zsum` are all equal to 0.
 * 7. If the condition is true, print "YES".
 * 8. Otherwise, print "NO".
 *
 * @implementation_note
 * This implementation correctly calculates the sums in a single pass. However, it
 * declares arrays `x`, `y`, and `z` to store all the vector components, which are
 * then never used again after being added to the sums. A more memory-efficient
 * implementation would omit these arrays and use simple temporary variables within
 * the loop to read the components.
 *
 * @complexity
 * Time Complexity: O(n), for the single loop through the `n` vectors.
 * Space Complexity: O(1) if we ignore the redundant arrays. As written, it's O(101)
 * due to the fixed-size arrays, which is constant space.
 */
#include <iostream>
using namespace std;

int main() {
  // `x`, `y`, `z` are fixed-size arrays to store vector components.
  // Note: Their size is 101, assuming n <= 100.
  int x[101], y[101], z[101];
  int n;
  
  // Initialize sums for each component axis.
  int xsum = 0, ysum = 0, zsum = 0;
  
  // Read the number of vectors.
  cin >> n;
  
  /**
   * @brief Loop to read vector components and calculate sums simultaneously.
   */
  for (int i = 0; i < n; i++) {
    // Read the components of the i-th vector.
    cin >> x[i] >> y[i] >> z[i];
    
    // Add the components to the running totals.
    xsum += x[i];
    ysum += y[i];
    zsum += z[i];
  }
  
  // Check for the equilibrium condition.
  if (xsum == 0 && ysum == 0 && zsum == 0)
    cout << "YES";
  else
    cout << "NO";
    
  return 0;
}
