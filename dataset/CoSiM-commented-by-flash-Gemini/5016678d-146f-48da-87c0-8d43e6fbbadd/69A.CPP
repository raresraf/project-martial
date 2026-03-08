/**
 * @file 69A.CPP
 * @brief Determines if a system of force vectors is in equilibrium.
 *
 * Problem Name: Young Physicist (Codeforces 69A)
 * Algorithm: Iterative summation of force components. Checks if the net force in x, y, and z directions is zero.
 * Time Complexity: O(N) where N is the number of force vectors.
 * Space Complexity: O(1) for storing the accumulated sums of components.
 */
#include <iostream> // Required for std::cin, std::cout.
using namespace std;

/**
 * @brief Main function to check if the system of forces is in equilibrium.
 * @pre Input consists of an integer 'n' (number of force vectors), followed by 'n' lines,
 *      each containing three integers (x, y, z) representing the components of a force vector.
 * @post Prints "YES" if the system is in equilibrium (all component sums are zero), otherwise "NO".
 */
main() { // Note: 'main' without return type is a C99 feature, typically 'int main()' in C++.
  int n; // Variable to store the number of force vectors.
  int sumx = 0, sumy = 0, sumz = 0; // Accumulators for the x, y, and z components.

  // Functional Utility: Reads the number of force vectors from standard input.
  cin >> n;

  /**
   * Block Logic: Iterates 'n' times to read each force vector and accumulate its components.
   * Invariant: In each iteration, 'x', 'y', and 'z' components of a force vector are read and added
   *            to their respective accumulators (`sumx`, `sumy`, `sumz`).
   */
  while (n > 0) { // Loop 'n' times.
    int x, y, z;  // Variables to store the components of the current force vector.
    // Functional Utility: Reads the three components of the current force vector.
    cin >> x >> y >> z;

    sumx += x; // Accumulates x-component.
    sumy += y; // Accumulates y-component.
    sumz += z; // Accumulates z-component.
    n -= 1;    // Decrements loop counter.
  }

  /**
   * Block Logic: Checks if the accumulated sums of all components are zero.
   * Pre-condition: `sumx`, `sumy`, `sumz` hold the total sums of x, y, and z components.
   * Invariant: If all three sums are zero, the system is in equilibrium.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES" << endl; // Prints "YES" if equilibrium.
  else
    cout << "NO" << endl; // Prints "NO" otherwise.
  // No explicit return 0; is needed for main in C++ if it reaches the end.
  // However, for competitive programming, it's often good practice to include it.
}
