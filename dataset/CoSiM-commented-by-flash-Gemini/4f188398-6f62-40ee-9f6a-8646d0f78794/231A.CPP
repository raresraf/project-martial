/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will implement, given judge votes.
 *        A problem is implemented if at least two out of three judges agree (vote '1').
 *
 * Problem Name: Young Physicist (Codeforces 231A)
 * Algorithm: Iterative counting of positive votes. For each problem, it tallies votes for agreement
 *            and increments a counter if the total agreement reaches a threshold of 2.
 * Time Complexity: O(N) where N is the number of problems, due to a single linear pass.
 * Space Complexity: O(1) as it uses a fixed number of variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, e.g., <iostream>.
using namespace std;

/**
 * @brief Main function to calculate the number of problems to be implemented.
 * @pre Input consists of an integer 'n' (total problems), followed by 'n' lines,
 *      each containing three binary integers (p, v, t) representing judge votes (0 or 1).
 * @post Prints the total count of problems that will be implemented.
 */
int main() {
  int i, n, cnt = 0, re = 0, p, v, t; // 'n': total problems, 'cnt': current problem's '1' votes,
                                     // 're': total implemented problems, 'p,v,t': judge votes.

  // Functional Utility: Reads the total number of problems.
  cin >> n;

  /**
   * Block Logic: Iterates through each problem to read judge votes and determine implementation.
   * Invariant: `re` accumulates the count of problems where `cnt` (votes for agreement) reaches 2 or more.
   *            `cnt` is reset for each new problem.
   */
  for (i = 0; i < n; i++) {
    // Functional Utility: Reads the three judge votes for the current problem.
    cin >> p >> v >> t;

    /**
     * Block Logic: Counts how many judges voted '1' for the current problem.
     * Pre-condition: `p`, `v`, `t` are binary (0 or 1).
     * Invariant: `cnt` stores the sum of '1' votes for the current problem.
     */
    if (p == 1) { // If first judge agrees.
      cnt++;
    }
    if (v == 1) { // If second judge agrees.
      cnt++;
    }
    if (t == 1) { // If third judge agrees.
      cnt++;
    }

    /**
     * Block Logic: Checks if the current problem will be implemented.
     * Pre-condition: `cnt` holds the number of '1' votes for the current problem.
     * Invariant: If `cnt` is 2 or more, `re` (total implemented problems) is incremented.
     */
    if (cnt >= 2) { // If at least two judges agreed.
      re++;         // Increment total implemented problems.
    }
    cnt = 0; // Resets 'cnt' for the next problem.
  }

  // Functional Utility: Prints the final count of problems to be implemented.
  cout << re;
  // Indicates successful program execution.
  return 0;
}
