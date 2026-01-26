/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem from Codeforces.
 *
 * This problem involves a team of three friends trying to solve programming problems.
 * For each problem, each friend indicates whether they are sure about the solution (1)
 * or not (0). The team decides to implement a problem if at least two friends
 * are sure about the solution. The task is to count how many problems the team will implement.
 *
 * Problem Link: https://codeforces.com/problemset/problem/231/A
 *
 * Algorithm: Iterative counting based on a simple threshold condition.
 * Time Complexity: O(N * K), where N is the number of problems and K is the
 *                  number of friends per problem (K=3, so effectively O(N)).
 * Space Complexity: O(1), as only a constant number of variables are used.
 */

#include <algorithm> // Generally useful, but not strictly needed for this specific problem.
#include <iostream>

using namespace std;

int main() {
  int n; // Declares an integer to store the number of problems.
  int count = 0; // Counter for problems the team will implement, initialized to 0.
  int line_count; // Stores the number of friends sure about the current problem.
  int temp; // Temporary variable to read each friend's certainty.

  // Reads the total number of problems `n` from standard input.
  cin >> n;

  /**
   * Block Logic: Processes each of the `n` problems.
   * Invariant: After each iteration, `count` accurately reflects the number
   *            of problems the team has decided to implement so far.
   */
  while (n--) { // Loop `n` times, once for each problem.
    line_count = 0; // Reset `line_count` for the current problem.
    
    /**
     * Block Logic: Reads the certainty votes of the three friends for the current problem.
     * Invariant: `line_count` accumulates the sum of certainty votes (0s and 1s).
     */
    for (int i = 0; i < 3; i++) { // There are always 3 friends.
      cin >> temp; // Reads a friend's vote (0 or 1).
      line_count += temp; // Adds the vote to `line_count`.
    }

    // Checks the condition for implementing the problem.
    // If at least two friends are sure (line_count >= 2).
    if (line_count >= 2) {
      count++; // Increment the total count of problems to be implemented.
    }
  }
  // Prints the final total number of problems the team will implement.
  cout << count << endl;
  return 0;
}