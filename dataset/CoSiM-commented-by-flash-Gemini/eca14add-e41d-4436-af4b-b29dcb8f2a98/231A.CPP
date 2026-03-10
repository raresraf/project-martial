/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem (Codeforces 231A).
 *   It counts the number of problems for which at least two out of three participants are sure about the solution.
 *   Problem description: https://codeforces.com/problemset/problem/231/A
 * * Algorithm: Iterative counting. It reads `n` sets of three integers (0 or 1),
 *   representing certainty votes. For each set, it sums these votes, and if the sum
 *   is greater than or equal to 2, it increments a total counter.
 * Time Complexity: O(N), where N is the number of problems, as it iterates through
 *   the problems once, performing constant time operations per problem.
 * Space Complexity: O(1) for a few integer variables.
 */
#include <algorithm> // Included but not directly used in the core logic.
#include <ctype.h>   // Included but not directly used.
#include <iostream>  // Required for std::cin, std::cout.
#include <map>       // Included but not directly used.
#include <stdio.h>   // Included but not directly used.
#include <string>    // Included but not directly used.

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
  cin >> n;
  int nr_problems = 0; // Accumulates the total count of problems where at least two participants are sure.
  /**
   * Block Logic: Iterates through each of the `n` problems.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: In each iteration, `aux1`, `aux2`, `aux3` represent the certainty votes for a problem.
   *            `nr_problems` accumulates problems where at least two participants are sure.
   * Inline: `auto i = 0` uses type inference, common in modern C++.
   */
  for (auto i = 0; i < n; i++) {
    int aux1, aux2, aux3; // `aux1`, `aux2`, `aux3`: Certainty votes (0 or 1) from three participants for the current problem.
    // Block Logic: Reads the three certainty votes for the current problem.
    cin >> aux1 >> aux2 >> aux3;
    /**
     * Conditional Logic: Checks if the sum of votes (`aux1 + aux2 + aux3`) is greater than or equal to 2.
     * If true, it means at least two participants are sure about the solution for this problem.
     */
    if (aux1 + aux2 + aux3 >= 2) {
      nr_problems++; // Increments the total counter of qualifying problems.
    }
  }
  // Block Logic: Prints the final total count of problems that meet the criteria.
  cout << nr_problems;
  return 0; // Indicates successful program termination.
}