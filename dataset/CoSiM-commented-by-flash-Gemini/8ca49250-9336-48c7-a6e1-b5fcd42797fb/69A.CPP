/**
 * @8ca49250-9336-48c7-a6e1-b5fcd42797fb/69A.CPP
 * @brief This program determines if a system of N three-dimensional force vectors
 * is in equilibrium. It calculates the sum of the x, y, and z components of all
 * forces. If all three resultant sums are zero, the system is in equilibrium.
 * Algorithm: Vector summation.
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1) for storing sums.
 */

#include <cstdio> // Required for C-style input/output functions (scanf, printf)

int main() {
  // @var x, y, z: Short integers to store the components of a single force vector.
  // @var n: Short integer to store the total number of force vectors.
  short x, y, z, n;
  // @var sumx, sumy, sumz: Integers to accumulate the sum of x, y, and z components, respectively.
  int sumx = 0, sumy = 0, sumz = 0;
  
  // Block Logic: Read the total number of force vectors.
  scanf("%hd", &n);
  
  // Block Logic: Iterate 'n' times, reading each force vector and accumulating its components.
  // Invariant: 'n' decreases with each iteration until all vectors are processed.
  // Pre-condition: 'n' specifies the number of force vectors to read.
  while (n--) {
    // Block Logic: Read the x, y, and z components of the current force vector.
    scanf("%hd %hd %hd", &x, &y, &z);
    // Inline: Accumulate the x-component.
    sumx += x;
    // Inline: Accumulate the y-component.
    sumy += y;
    // Inline: Accumulate the z-component.
    sumz += z;
  }
  
  // Block Logic: Check if the sum of all components is zero, indicating equilibrium.
  // Post-condition: Print "YES" if in equilibrium, "NO" otherwise.
  if (sumx == 0 && sumy == 0 && sumz == 0)
    printf("%s", "YES");
  else
    printf("%s", "NO");

  return 0; // Explicitly return 0 for successful execution.
}
