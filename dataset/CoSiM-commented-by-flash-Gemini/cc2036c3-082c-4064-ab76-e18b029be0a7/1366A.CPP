/**
 * @file 1366A.CPP
 * @brief Solution to the "Shovels and Swords" problem (Codeforces 1366A).
 *
 * This program calculates the maximum number of items that can be crafted given
 * two types of resources. Each item requires 1 unit of each resource, or 2 units
 * of one and 1 unit of another (implied by the (a+b)/3 constraint).
 * Algorithm: The maximum number of items is limited by three factors:
 *            1. The minimum available resource (min(a, b)).
 *            2. The total number of resource units divided by 3, as each item
 *               effectively uses 3 units of resources if the (2,1) crafting ratio
 *               is always possible and balanced.
 *            The solution takes the minimum of these two constraints.
 * Time Complexity: O(T) where T is the number of test cases, as each test case
 *                  involves constant time operations.
 * Space Complexity: O(1) - constant space used for variables.
 */

#include <bits/stdc++.h> // Includes various standard libraries for competitive programming.
using namespace std;       // Uses the standard namespace for convenience.

// Global variables for test cases (t), resources (a, b), and result (res).
int t, a, b, res;

/**
 * @brief Main function to solve the "Shovels and Swords" problem.
 *
 * Reads the number of test cases, and for each test case, reads resource counts
 * and calculates the maximum craftable items, then prints the result.
 * @return Implicitly 0 on successful execution as main() with no explicit return defaults to 0.
 */
main() { // Using 'main()' without explicit return type, which is implicitly 'int' in C++.
  // Block Logic: Read the number of test cases.
  // Pre-condition: User provides a single integer 't'.
  // Invariant: 't' holds the count of test cases.
  cin >> t;

  // Block Logic: Process each test case.
  // Invariant: 't' decrements with each iteration, loop continues until all test cases are processed.
  while (t--) {
    res = 0; // Invariant: 'res' is reset for each test case before calculation.

    // Block Logic: Read the counts of the two resources, 'a' and 'b'.
    // Pre-condition: User provides two integers for each test case.
    // Invariant: 'a' and 'b' hold the resource counts.
    cin >> a >> b;

    // Block Logic: Calculate the maximum number of items that can be crafted.
    // The number of items is limited by:
    // 1. The resource that is less abundant (min(a, b)).
    // 2. The total resources divided by 3, considering the crafting cost ( (a + b) / 3 ).
    // The final result is the minimum of these two limiting factors.
    // Post-condition: The maximum craftable items for the current test case is printed.
    cout << min(min(a, b), (a + b) / 3) << '\n';
  }
}
