/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A: Team (Array-based Implementation).
 * @details This program processes 'n' problem statements, each with three opinions (represented by integers, typically 0 or 1).
 * It determines how many problems will be implemented. A problem is implemented if at least two people agree
 * that it should be implemented, which is robustly checked by comparing combinations of opinions in an array.
 * @algorithm Iterative processing with direct conditional checks.
 * Time Complexity: $O(N)$ where N is the number of problem statements.
 * Space Complexity: $O(N \cdot K)$ where K is the number of opinions per problem (constant 3 here), to store opinions in a 2D array.
 */
#include <iostream>
using namespace std;
/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A: Team (Array-based Implementation).
 * @details This program processes 'n' problem statements, each with three opinions (represented by integers, typically 0 or 1).
 * It determines how many problems will be implemented. A problem is implemented if at least two people agree
 * that it should be implemented, which is robustly checked by comparing combinations of opinions in an array.
 * @algorithm Iterative processing with direct conditional checks.
 * Time Complexity: $O(N)$ where N is the number of problem statements.
 * Space Complexity: $O(N \cdot K)$ where K is the number of opinions per problem (constant 3 here), to store opinions in a 2D array.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).
using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the problem counting logic.
   */
  int n, // @var n: The total number of problem statements to process.
      i, // @var i: Loop counter for iterating through problems.
      j, // @var j: Loop counter for iterating through opinions within a problem.
      k, // @var k: Declared but unused.
      l = 0; // @var l: Counter for the number of problems that will be implemented.

  cin >> n; // Reads the total number of problem statements.

  // Functional Utility: Declares a 2D array to store opinions for each problem.
  // Note: Using Variable Length Arrays (VLAs) which are a GCC extension and not standard C++.
  int a[n][3]; // @var a: A 2D array where `a[i][0]`, `a[i][1]`, `a[i][2]` store opinions for problem `i`.

  /**
   * @brief Reads the opinions for all 'n' problem statements.
   * @block_logic Nested loops iterate through each problem and its three opinions.
   * @invariant `a[i][j]` stores the opinion of the j-th person for the i-th problem.
   */
  for (i = 0; i < n; i++)
    for (j = 0; j < 3; j++)
      cin >> a[i][j]; // Reads opinions into the 2D array.

  /**
   * @brief Checks each problem statement to see if it meets the implementation criteria.
   * @block_logic Iterates through each problem stored in array `a`.
   * @invariant `l` accumulates the count of problems meeting the criteria.
   */
  for (i = 0; i < n; i++) {
    /**
     * @brief Conditional check: at least two opinions for the current problem are '1'.
     * @details This logical expression checks all three pairs of opinions to see if both
     *          opinions in any pair are '1'.
     * @block_logic If any of the three paired conditions are true, the problem is counted.
     * @invariant `l` is incremented if the current problem meets the "at least two '1's" criteria.
     */
    if ((a[i][0] == 1 && a[i][1] == 1) || // Checks if opinion 0 AND opinion 1 are '1'.
        (a[i][2] == 1 && a[i][1] == 1) || // Checks if opinion 2 AND opinion 1 are '1'.
        (a[i][0] == 1 && a[i][2] == 1))   // Checks if opinion 0 AND opinion 2 are '1'.
      l++; // Increments the counter for implemented problems.
  }

  // Functional Utility: Prints the total count of problems that will be implemented.
  cout << l;
  return 0; // Indicates successful execution.
}
