/**
 * @file 1366A.CPP
 * @brief Solution to Codeforces Problem 1366A: "Shovels and Swords".
 *
 * This problem involves crafting items (shovels or swords) from iron (`a`) and diamond (`b`)
 * resources. A shovel costs 2 iron and 1 diamond. A sword costs 1 iron and 2 diamonds.
 * The goal is to maximize the total number of items crafted.
 *
 * Algorithm: The maximum number of items is limited by two factors:
 * 1. The total amount of resources: each item requires 3 units of material (2+1 or 1+2). So, at most `(a + b) / 3` items can be crafted.
 * 2. The minimum resource available: each item requires at least 1 iron and 1 diamond.
 *    If we always ensure `a <= b` by swapping, then the number of items is also limited by `a`.
 * Combining these, the maximum items is `min((a + b) / 3, a)`.
 *
 * Time Complexity: O(T), where T is the number of test cases, as each test case involves
 *                  a constant number of arithmetic operations and comparisons.
 * Space Complexity: O(1) as only a few variables are used per test case.
 */
#include <iostream> // Required for input/output operations (cin, cout, endl).
#include <algorithm> // Required for `std::swap`.
using namespace std;

/**
 * @brief Main function to solve the "Shovels and Swords" problem.
 * Reads the number of test cases, then for each test case, reads the available
 * iron and diamond and calculates the maximum number of items that can be crafted.
 */
int main() {
  int t; // Stores the number of test cases.
  long long a, b; // 'a': units of iron, 'b': units of diamond. Using long long to prevent overflow.
  cin >> t; // Reads the number of test cases.

  /**
   * Block Logic: Processes each test case one by one.
   * Pre-condition: `t` specifies the number of scenarios to run.
   * Invariant: After each iteration, one test case has been solved and its result printed.
   */
  while (t--) {
    cin >> a >> b; // Reads the available iron and diamond.
    long long ans = 0; // Stores the maximum number of items that can be crafted.
    
    // Ensures that 'a' is always less than or equal to 'b'.
    // This simplifies the logic by making 'a' effectively the limiting resource if considering the `min(a, b)` aspect.
    if (a > b) {
      swap(a, b);
    }
    
    // Calculate the maximum number of items possible.
    // This is limited by the total resources (each item costs 3 units) and also by the minimum resource available ('a').
    ans = (a + b) / 3; // Initial upper bound based on total materials.
    
    /**
     * Block Logic: Refines the maximum items based on the minimum resource available.
     * Ensures that the number of items does not exceed the count of the scarcer resource ('a').
     * Effectively, `ans = min(ans, a)`.
     */
    if (a < ans) {
      ans = a; // If 'a' is scarcer than the total material limit allows, 'a' becomes the new limit.
    }
    
    cout << ans << endl; // Prints the maximum number of items that can be crafted.
  }
  return 0; // Indicates successful execution.
}
