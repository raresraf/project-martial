/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" competitive programming problem.
 * @details The problem is to determine if a body is in equilibrium under the influence
 * of several force vectors in 3D space. A body is in equilibrium if and only if the
 * vector sum of all forces acting on it is the zero vector, i.e., (0, 0, 0).
 *
 * @section algorithm Algorithm
 * The algorithm directly implements the principle of equilibrium from physics:
 * 1. Read the number of applied force vectors, `st`.
 * 2. Initialize three variables, `x`, `y`, and `z`, to zero. These will store the
 *    sum of the components of all force vectors along the X, Y, and Z axes.
 * 3. Loop `st` times, once for each force vector.
 * 4. In each iteration, read the three components (`xx`, `yy`, `zz`) of a force vector.
 * 5. Add each component to its corresponding sum (`x`, `y`, `z`). The code uses a
 *    custom `subsadd` function for this, which is an unnecessarily complex way to
 *    perform simple addition.
 * 6. After the loop, check if the sums of all three components are zero.
 * 7. If `x`, `y`, and `z` are all zero, it means the net force is the zero vector,
 *    and the body is in equilibrium. The program prints "YES".
 * 8. Otherwise, the net force is non-zero, and the program prints "NO".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of force vectors, due to the single loop that processes each vector once.
 * - Space Complexity: O(1), as a fixed amount of memory is used regardless of the input size.
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h> // For abs()
using namespace std;

/**
 * @brief An unnecessarily complex function to perform addition of two integers.
 * @param x The base number.
 * @param y The number to add to x.
 * @return The sum of x and y.
 * @note This function is functionally equivalent to a simple `x + y` operation.
 */
int subsadd(int x, int y) {
  if (y < 0)
    x -= abs(y); // This is the same as x + y when y is negative.
  else
    x += y; // This is x + y when y is non-negative.
  return x;
};

int main() {
  int st;      // Represents the number of force vectors.
  int x, y, z; // Used to store the sum of vector components along X, Y, and Z axes.
  x = 0;
  y = 0;
  z = 0;
  
  cin >> st;
  
  // Loop through each force vector and add its components to the total sum.
  for (int i = 0; i < st; i++) {
    int xx, yy, zz; // Components of the current force vector.
    cin >> xx >> yy >> zz;
    
    // Add the components to their respective running sums.
    x = subsadd(x, xx);
    y = subsadd(y, yy);
    z = subsadd(z, zz);
  }
  
  // After summing all vectors, check if the resultant vector is (0, 0, 0).
  if (x == 0 && y == 0 && z == 0)
    cout << "YES"; // The body is in equilibrium.
  else
    cout << "NO"; // The body is not in equilibrium.
    
  return 0;
}