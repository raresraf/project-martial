/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A: Team.
 * @details This program processes 'n' problem statements, each with three opinions (represented by integers a, b, c).
 * It determines how many problems will be implemented. A problem is implemented if at least two people agree
 * that it should be implemented, which is interpreted as at least two of the three opinion values are non-zero.
 * The solution uses bitwise AND operations to check for non-zero conditions, which simplifies the logical AND.
 * @algorithm Counting based on logical conditions.
 * Time Complexity: $O(N)$ where N is the number of problem statements.
 * Space Complexity: $O(1)$.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  /**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A: Team.
 * @details This program processes 'n' problem statements, each with three opinions (represented by integers a, b, c).
 * It determines how many problems will be implemented. A problem is implemented if at least two people agree
 * that it should be implemented, which is interpreted as at least two of the three opinion values are non-zero.
 * The solution uses bitwise AND operations to check for non-zero conditions, which simplifies the logical AND.
 * @algorithm Counting based on logical conditions.
 * Time Complexity: $O(N)$ where N is the number of problem statements.
 * Space Complexity: $O(1)$.
 */
#include <bits/stdc++.h> // Standard C++ library includes (e.g., iostream, vector, algorithm).

using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the problem solving logic.
   */

  int n;     // @var n: The total number of problem statements.
  int a, b, c; // @var a, b, c: Integers representing the opinions of three individuals for a problem.
  int count = 0; // @var count: Counter for the number of problems that will be implemented.

  scanf("%d", &n); // Reads the total number of problem statements from standard input.

  /**
   * @brief Processes each of the 'n' problem statements.
   * @block_logic Iterates 'n' times, processing one problem statement in each iteration.
   * @invariant `n` decreases by 1 in each iteration until it becomes 0.
   */
  while (n--) {
    // Functional Utility: Reads the three opinions for the current problem statement.
    scanf("%d%d%d", &a, &b, &c);

    /**
     * @brief Checks if at least two out of three opinions are non-zero using bitwise AND.
     * @details The condition `a & b || b & c || c & a` evaluates to true if the bitwise AND
     *          of any two pairs of (a, b), (b, c), or (c, a) results in a non-zero value.
     *          In many competitive programming scenarios with positive integers acting as boolean
     *          flags (0 for false, non-zero for true), this expression effectively checks if
     *          at least two of `a`, `b`, `c` are non-zero.
     * @block_logic Increments `count` if the condition is met.
     * @invariant `count` stores the number of problems where the bitwise condition is satisfied.
     */
    if (a & b || b & c || c & a)
      count++;
  }

  // Functional Utility: Prints the total count of problems that will be implemented to standard output.
  cout << count << endl;
  return 0; // Indicates successful execution.
}