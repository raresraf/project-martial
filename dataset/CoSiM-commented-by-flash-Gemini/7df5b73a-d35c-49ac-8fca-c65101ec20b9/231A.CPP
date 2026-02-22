/**
 * @file 231A.CPP
 * @brief Solution attempt for Codeforces Problem 231A: Team (Flawed Implementation).
 * @details This program aims to count the number of problems that will be implemented,
 * where a problem is implemented if at least two people agree (represented by an opinion of '1').
 * The implementation attempts to count combinations of opinions using a series of conditional checks
 * but contains logical flaws in its counting mechanism and control flow, which may lead to incorrect results.
 * It reads opinions `a, b, c` for 'j' problems.
 * @algorithm Conditional counting with logical inconsistencies.
 * Time Complexity: $O(N)$ where N is the total number of problems across all test cases.
 * Space Complexity: $O(1)$ to store a few variables.
 * @warning This implementation contains logical errors and may not correctly solve the problem.
 */
#include <cmath>
#include <iostream>
using namespace std;
/**
 * @file 231A.CPP
 * @brief Solution attempt for Codeforces Problem 231A: Team (Flawed Implementation).
 * @details This program aims to count the number of problems that will be implemented,
 * where a problem is implemented if at least two people agree (represented by an opinion of '1').
 * The implementation attempts to count combinations of opinions using a series of conditional checks
 * but contains logical flaws in its counting mechanism and control flow, which may lead to incorrect results.
 * It reads opinions `a, b, c` for 'j' problems.
 * @algorithm Conditional counting with logical inconsistencies.
 * Time Complexity: $O(N)$ where N is the total number of problems across all test cases.
 * Space Complexity: $O(1)$ to store a few variables.
 * @warning This implementation contains logical errors and may not correctly solve the problem.
 */
#include <cmath>    // Included but not directly used for core logic.
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the problem counting logic.
   */
  long long int a, b, c, // @var a, b, c: Integers representing opinions for a problem.
                i, j,    // @var i: Loop counter. @var j: Number of problem instances to follow (per input read).
                key = 0, // @var key: Counter, intended to track certain combinations of '1' opinions.
                cot = 0, // @var cot: Counter, intended to track specific combinations of '1' opinions.
                sum;     // @var sum: Declared but not initialized or used meaningfully in the counting logic.
  long long vim;         // @var vim: Declared but not initialized or used in the counting logic.

  /**
   * @brief Main loop for processing multiple sets of problems (test cases).
   * @block_logic Reads an integer `j` (interpreted as number of problems) and processes problems until End-Of-File.
   * @invariant Continues until no more input for `j` is available.
   * @note The problem interpretation for `j` is ambiguous, as it is used both as a loop limit and input.
   */
  while (!(cin >> j).eof()) { // Reads the number of problems 'j' for the current set. Checks for EOF.
    /**
     * @brief Loop for processing 'j' problem instances.
     * @block_logic Reads three opinions for each problem and attempts to update counters.
     * @invariant `i` increments, `a, b, c` are read, `key` and `cot` are updated.
     */
    for (i = 1; i <= j; i++) {
      cin >> a >> b >> c; // Reads the three opinions for the current problem.

      /**
       * @brief Conditional checks for various combinations of '1' opinions.
       * @details These `if-else if` blocks attempt to identify scenarios where opinions `a`, `b`, and `c`
       *          are '1'. However, the logic `a && b == 1` interprets as `(a != 0) && (b == 1)`,
       *          which is generally not equivalent to checking if both `a` and `b` are '1'.
       *          The specific combinations and their impact on `key` and `cot` are logically inconsistent
       *          with the common interpretation of "at least two 1s" for the problem.
       * @block_logic Increments `key` or `cot` based on specific (and potentially flawed) conditions.
       * @invariant `key` and `cot` reflect the counts according to the executed `if-else if` branches.
       */
      if (a && b == 1) { // Checks if 'a' is non-zero AND 'b' is 1.
        key++;           // Increments `key`.
      } else if (b && c == 1) { // Checks if 'b' is non-zero AND 'c' is 1.
        key++;           // Increments `key`.
      } else if (a && c == 1) { // Checks if 'a' is non-zero AND 'c' is 1.
        key++;           // Increments `key`.
      } else if (a && b && c == 1) { // Checks if 'a' is non-zero AND 'b' is non-zero AND 'c' is 1.
                                     // This branch is likely unreachable if previous conditions with `== 1` are met.
        cot++;           // Increments `cot`.
      } else if (a || b || c == 1) { // Checks if 'a' is non-zero OR 'b' is non-zero OR 'c' is 1.
        sum = 0;           // Resets `sum` (but `sum` is not used in output).
      }
    }
    // Functional Utility: The commented line `vim=key+cot+sum;` suggests an attempt to aggregate, but `vim` is unused.
    // vim=key+cot+sum;
    
    // Functional Utility: Prints the sum of `key` and `cot` (the result of the flawed counting).
    cout << key + cot << endl;
    
    // Functional Utility: Resets counters for the next set of problems.
    key = 0; // Resets `key`.
    cot = 0; // Resets `cot`.
  }
}
