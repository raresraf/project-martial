/**
 * @5a9cf1ce-80f6-4ba3-b231-27c5cc50d8ed/69A.CPP
 * @brief This program solves problem 69A, which involves determining if a system
 * of 'n' three-dimensional vectors is in equilibrium. This is achieved by
 * checking if the sum of all x-components, y-components, and z-components
 * across all vectors are simultaneously zero.
 *
 * Algorithm: Linear scan to read vector components and sum them, followed by a conditional check.
 * Time Complexity: O(N) where N is the number of vectors.
 * Space Complexity: O(1) for storing variables.
 */

#include <iostream> // Included but uses C-style I/O functions.
#include <cstdio>   // For scanf and printf.

using namespace std; // Uses the standard namespace.

int main() {
  int n; // Declares 'n', representing the number of 3D vectors.
  // Declares x, y, z for individual vector components, and sx, sy, sz for accumulating sums.
  int x, y, z; 
  int sx = 0, sy = 0, sz = 0; // Initializes component sums to zero.
  
  scanf("%d", &n); // Reads the number of vectors 'n' from standard input.
  
  /**
   * Block Logic: Processes 'n' 3D vectors. For each vector, it reads its components
   * and adds them to the respective cumulative sums.
   * Invariant: After this loop, 'sx', 'sy', 'sz' hold the resultant sums of all x, y, and z components.
   */
  while (n--) { // Loop 'n' times, decrementing 'n' in each iteration.
    // Reads the three components (x, y, z) of the current vector from standard input.
    scanf("%d %d %d", &x, &y, &z); 
    sx += x; // Inline: Adds the x-component to the total sum of x-components.
    sy += y; // Inline: Adds the y-component to the total sum of y-components.
    sz += z; // Inline: Adds the z-component to the total sum of z-components.
  }
  
  /**
   * Block Logic: Checks if the system is in equilibrium.
   * Pre-condition: 'sx', 'sy', 'sz' contain the final resultant sums.
   * Invariant: Prints "YES" if all sums are zero, otherwise "NO".
   */
  if (sx == 0 && sy == 0 && sz == 0) { // If all three resultant sums are zero.
    printf("YES"); // Prints "YES".
  } else {
    printf("NO"); // Prints "NO".
  }
  // The program implicitly returns 0 on successful completion of main.
}