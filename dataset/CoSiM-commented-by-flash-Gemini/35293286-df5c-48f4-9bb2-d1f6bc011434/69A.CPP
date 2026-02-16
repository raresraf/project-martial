/**
 * @file 69A.CPP
 * @brief Solution for a competitive programming problem that determines if a system of vectors is in equilibrium.
 *
 * Algorithm: Reads 'n' 3D vectors. It calculates the sum of their x, y, and z components independently.
 *            The system is in equilibrium if and only if all three sums (sumX, sumY, sumZ) are zero.
 * Time Complexity: O(N) where N is the number of vectors. Each vector involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) as it uses a few constant-size variables for sums.
 */

#include <iostream>
using namespace std;

/**
 * @brief Main entry point of the program.
 *
 * Functional Utility: Reads an integer `n` representing the number of 3D vectors.
 *                     It then reads `n` vectors, accumulates their component-wise
 *                     sums, and finally determines if the sum of all vectors is
 *                     the zero vector (i.e., the system is in equilibrium).
 *
 * @return 0 on successful execution.
 */
int main() {
  int n; // @brief Number of 3D vectors to be read.
  int x, y, z; // @brief Components of the current 3D vector.
  int sumx = 0; // @brief Accumulator for the sum of x-components.
  int sumy = 0; // @brief Accumulator for the sum of y-components.
  int sumz = 0; // @brief Accumulator for the sum of z-components.
  cin >> n;
  // Block Logic: Iterate `n` times to read each 3D vector and accumulate its components.
  // Invariant: After each iteration, `sumx`, `sumy`, and `sumz` hold the accumulated
  //            sums of the components of all vectors processed so far.
  while (n--) {
    cin >> x >> y >> z; // Functional Utility: Read the x, y, and z components of the current vector.
    sumx += x; // Functional Utility: Add current x-component to `sumx`.
    sumy += y; // Functional Utility: Add current y-component to `sumy`.
    sumz += z; // Functional Utility: Add current z-component to `sumz`.
  }
  // Block Logic: Check if the system of vectors is in equilibrium.
  // Invariant: The system is in equilibrium if and only if the sum of all x, y,
  //            and z components is exactly zero.
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES"; // Functional Utility: Output "YES" if the system is in equilibrium.
  else
    cout << "NO"; // Functional Utility: Output "NO" if the system is not in equilibrium.

  return 0;
}
