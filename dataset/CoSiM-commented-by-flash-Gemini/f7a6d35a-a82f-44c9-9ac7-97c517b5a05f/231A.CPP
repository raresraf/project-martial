/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem (Codeforces 231A).
 *   It counts the number of problems for which at least two out of three participants are sure about the solution.
 *   Problem description: https://codeforces.com/problemset/problem/231/A
 * * Algorithm: Iterative counting. It reads `n` sets of three integers (0 or 1),
 *   representing certainty votes. For each set, it sums these votes, and if the sum
 *   is greater than 1 (meaning at least two votes are 1), it increments a total counter.
 * Time Complexity: O(N), where N is the number of problems, as it iterates through
 *   the problems once, performing constant time operations per problem.
 * Space Complexity: O(1) for a few integer variables.
 */
#include <iostream> // Required for standard input/output operations.

using namespace std;

/**
 * @brief Main function to execute the problem counting logic.
 * Functional Utility: Reads inputs for multiple problems, processes them to count
 *                     sure participants, and outputs the total number of problems
 *                     meeting the criteria.
 */
int main() {
  int n; // Stores the total number of problems.
  // Block Logic: Reads the total number of problems, `n`.
  std::cin >> n;
  int count = 0; // Accumulates the total count of problems where at least two participants are sure.
  /**
   * Block Logic: Iterates `n` times to process each problem.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: In each iteration, `a`, `b`, `c` represent the certainty votes for a problem.
   *            `count` accumulates problems where at least two participants are sure.
   * Inline: `n--` decrements the counter and evaluates to true as long as `n` is greater than 0.
   */
  while (n--) {
    int a, b, c; // `a`, `b`, `c`: Certainty votes (0 or 1) from three participants for the current problem.
    // Block Logic: Reads the three certainty votes for the current problem.
    std::cin >> a >> b >> c;
    /**
     * Conditional Logic: Checks if the sum of votes (`a + b + c`) is greater than 1.
     * If true, it means at least two participants are sure about the solution for this problem.
     */
    if ((a + b + c) > 1)
      count++; // Increments the total counter of qualifying problems.
  }
  // Block Logic: Prints the final total count of problems that meet the criteria.
  std::cout << count;
  return 0; // Indicates successful program termination.
}