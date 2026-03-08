/**
 * @file 69A.CPP
 * @brief Determines if a system of force vectors is in equilibrium.
 *
 * Problem Name: Young Physicist (Codeforces 69A)
 * Algorithm: Iterative summation of force components. Checks if the net force in x, y, and z directions is zero.
 * Time Complexity: O(N) where N is the number of force vectors.
 * Space Complexity: O(1) for storing the accumulated sums of components.
 */
#include <algorithm> // Not directly used in this code, but included in competitive programming templates.
#include <deque>       // Not directly used.
#include <iostream>    // Required for input/output operations (cin, cout).
#include <map>         // Not directly used.
#include <queue>       // Not directly used.
#include <set>         // Not directly used.
#include <stack>       // Not directly used.
#include <string>      // Not directly used.
#include <unordered_map> // Not directly used.
#include <unordered_set> // Not directly used.
#include <vector>      // Required for std::vector.

#define ull unsigned long long // Functional Utility: Defines 'ull' as a shorthand for 'unsigned long long'.
#define ll long long         // Functional Utility: Defines 'll' as a shorthand for 'long long'.

using namespace std;

/**
 * @brief Main function to check if the system of forces is in equilibrium.
 * @pre Input consists of an integer 'n' (number of force vectors), followed by 'n' lines,
 *      each containing three integers (x, y, z) representing the components of a force vector.
 * @post Prints "YES" if the system is in equilibrium (all component sums are zero), otherwise "NO".
 */
int main() {
  int n; // Variable to store the number of force vectors.
  // Functional Utility: Reads the number of force vectors from standard input.
  cin >> n;

  vector<int> sum(3); // Vector to store the accumulated sums of x, y, and z components.
                      // sum[0] for x-components, sum[1] for y-components, sum[2] for z-components.

  /**
   * Block Logic: Iterates 'n' times to read each force vector and accumulate its components.
   * Invariant: In each iteration, 'x', 'y', and 'z' components of a force vector are read and added
   *            to their respective accumulators in the 'sum' vector.
   */
  while (n--) {
    int x, y, z; // Variables to store the components of the current force vector.
    // Functional Utility: Reads the three components of the current force vector.
    cin >> x >> y >> z;

    sum[0] += x; // Accumulates x-component.
    sum[1] += y; // Accumulates y-component.
    sum[2] += z; // Accumulates z-component.
  }

  /**
   * Block Logic: Checks if the accumulated sums of all components are zero.
   * Pre-condition: 'sum' vector contains the total sums of x, y, and z components.
   * Invariant: If all three sums (sum[0], sum[1], sum[2]) are zero, the system is in equilibrium.
   */
  cout << (sum[0] == 0 && sum[1] == 0 && sum[2] == 0 ? "YES" : "NO"); // Prints "YES" or "NO".
  // Indicates successful program execution.
  return 0;
}