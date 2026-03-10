/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem (Codeforces 231A).
 *   It counts the number of problems for which at least two out of three participants are sure about the solution.
 *   Problem description: https://codeforces.com/problemset/problem/231/A
 * * Algorithm: Iterative counting. It reads `n` sets of three integers (0 or 1),
 *   representing certainty votes. For each set, it sums these votes, and if the sum
 *   is 2 or more, it increments a total counter.
 * Time Complexity: O(N), where N is the number of problems, as it iterates through
 *   the problems once, performing constant time operations per problem.
 * Space Complexity: O(1) for a few integer variables.
 */
//
// Created by Taro Young on 2020/9/6.
//
#include <iostream> // Required for standard input/output operations.

using namespace std;

/**
 * @brief Main function to execute the problem counting logic.
 * Functional Utility: Reads inputs for multiple problems, processes them to count
 *                     sure participants, and outputs the total number of problems
 *                     meeting the criteria.
 */
int main() {
  int n;   // Stores the total number of problems.
  int num = 0; // Accumulates the total count of problems where at least two participants are sure.
  // Block Logic: Reads the total number of problems, `n`.
  cin >> n;
  /**
   * Block Logic: Iterates through each of the `n` problems.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: In each iteration, `P`, `V`, `T` represent the certainty votes for a problem.
   *            `num` accumulates problems where at least two participants are sure.
   */
  for (int i = 0; i < n; i++) {
    int P, V, T; // `P`, `V`, `T`: Certainty votes (0 or 1) from three participants for the current problem.
    // Block Logic: Reads the three certainty votes for the current problem.
    cin >> P >> V >> T;
    /**
     * Conditional Logic: Checks if the sum of votes (`P + V + T`) is 2 or more.
     * If true, it means at least two participants are sure about the solution for this problem.
     */
    if (P + V + T >= 2)
      num++; // Increments the total counter of qualifying problems.
  }
  // Block Logic: Prints the final total count of problems that meet the criteria.
  cout << num << endl;
  return 0;
}
