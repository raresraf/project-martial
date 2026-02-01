/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces contest 69.
 *
 * @details
 * This program determines if a body is in equilibrium under the influence of several forces.
 * According to Newton's first law, a body is in equilibrium if the vector sum of all forces
 * acting upon it is zero. In a 3D Cartesian coordinate system, this means that the sum of the
 * x-components, the sum of the y-components, and the sum of the z-components of all force
 * vectors must each independently be zero.
 *
 * The program reads an integer `n`, which is the number of force vectors. It then reads `n`
 * vectors, each represented by its three integer components (x, y, z). It maintains three
 * running sums for the x, y, and z components respectively. After reading all the vectors,
 * it checks if all three sums are equal to zero.
 *
 * Algorithm:
 * 1. Initialize sums for x, y, and z components to zero (`sum1`, `sum2`, `sum3`).
 * 2. Read the number of force vectors, `n`.
 * 3. Loop `n` times:
 *    a. Read the x, y, and z components of the current force vector.
 *    b. Add the x-component to `sum1`.
 *    c. Add the y-component to `sum2`.
 *    d. Add the z-component to `sum3`.
 * 4. After the loop, check if `sum1`, `sum2`, and `sum3` are all zero.
 * 5. If they are all zero, print "YES". Otherwise, print "NO".
 *
 * Time Complexity: O(n), where `n` is the number of force vectors, because we iterate through the vectors once.
 * Space Complexity: O(1), as we only use a constant amount of space for variables regardless of the input size.
 */
#include <stdio.h>
int main() {
  int a, i, m, n, o, sum1 = 0, sum2 = 0, sum3 = 0;
  // Read the total number of force vectors.
  scanf("%d", &a);
  
  // Block Logic: Iterate 'a' times to read each force vector and update the component sums.
  // Invariant: After each iteration, sum1, sum2, and sum3 hold the cumulative sum of x, y, and z components
  // for the forces processed so far.
  for (i = 0; i < a; i++) {
    // Read the three components of the current force vector.
    scanf("%d", &m); // x-component
    sum1 += m;
    scanf("%d", &n); // y-component
    sum2 += n;
    scanf("%d", &o); // z-component
    sum3 += o;
  }
  
  // Pre-condition: Check if the net force vector is the zero vector.
  // This is the condition for static equilibrium.
  if (sum1 == 0 && sum2 == 0 && sum3 == 0)
    printf("YES\n");
  else
    printf("NO\n");
}