/**
 * @file 69A.CPP
 * @brief Solution to a problem determining if a system of force vectors is in equilibrium.
 *
 * This program reads an integer 'n', representing the number of 3D vectors.
 * For each of the 'n' vectors, it reads its three components (x, y, z).
 * It then calculates the sum of all x-components, all y-components, and all z-components.
 * The problem statement implies that for the system to be in equilibrium, the sum of
 * all components across all vectors must be zero. If this condition is met, it prints "YES";
 * otherwise, it prints "NO".
 *
 * Algorithm: Iterative summation of all x, y, and z components of 'n' vectors into a single accumulator.
 * Time Complexity: O(N), where N is the number of vectors, as each vector is processed once.
 * Space Complexity: O(1) for variable storage.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Declare 'n' for the number of vectors, and 'ans' initialized to 0 to store the total sum of components.
  int n, ans = 0;
  // Read the total number of vectors 'n'.
  scanf("%d", &n);
  /**
   * Block Logic: Loop through each of the 'n' vectors to read their components and sum them up.
   * Invariant: 'n' decreases with each iteration; 'ans' accumulates the sum of components.
   */
  while (n--) {
    // Declare 'a', 'b', 'c' for the x, y, and z components of the current vector.
    int a, b, c;
    // Read the three components of the current 3D vector.
    scanf("%d %d %d", &a, &b, &c);
    // Add the sum of the current vector's components to the total sum 'ans'.
    ans += a + b + c;
  }
  /**
   * Block Logic: Checks if the total sum of all components is zero to determine equilibrium.
   * Pre-condition: 'ans' holds the aggregated sum of all x, y, and z components.
   */
  if (ans == 0)
    // If the total sum is zero, the system is in equilibrium.
    printf("YES\n");
  else
    // Otherwise, the system is not in equilibrium.
    printf("NO\n");
  return 0; // Indicate successful program execution.
}
