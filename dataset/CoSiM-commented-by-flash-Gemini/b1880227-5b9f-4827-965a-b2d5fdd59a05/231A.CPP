/**
 * @file 231A.CPP
 * @brief This program determines the number of problems that can be implemented.
 * For each problem, three friends provide their opinion (1 if sure, 0 if not sure).
 * A problem is implemented if at least two friends are sure about the solution.
 *
 * Algorithm: Iterative problem evaluation based on a simple majority rule.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1)
 */

#include <bits/stdc++.h> // Includes iostream, etc.
using namespace std;

int main() {
  // Functional Utility: 'N' stores the total number of problems to evaluate.
  int N;
  // Functional Utility: 'cnt' temporarily counts the number of friends who are sure for the current problem.
  int cnt = 0;
  // Functional Utility: 'dm' (decision/mark) temporarily stores a friend's opinion (1 or 0).
  int dm;
  // Functional Utility: 'res' (result) accumulates the total count of implementable problems.
  int res = 0;

  // Block Logic: Reads the total number of problems 'N' from standard input.
  cin >> N;

  /**
   * Block Logic: Iterates through each problem to evaluate if it can be implemented.
   * Invariant: At the start of each iteration, 'i' represents the number of problems already evaluated.
   */
  for (int i = 1; i <= N; i++) {
    // Functional Utility: Resets 'cnt' for each new problem.
    cnt = 0;
    /**
     * Block Logic: Reads the opinions of the three friends for the current problem.
     * Invariant: After each iteration, 'dm' holds an opinion, and 'cnt' is updated if the opinion is '1'.
     */
    for (int j = 1; j <= 3; j++) {
      // Block Logic: Reads a friend's opinion.
      cin >> dm;
      /**
       * Block Logic: Checks if the current friend is sure (opinion is '1').
       * Pre-condition: 'dm' contains the current friend's opinion.
       * Invariant: 'cnt' accurately reflects the number of sure friends so far for this problem.
       */
      if (dm == 1)
        cnt++;
    }
    /**
     * Block Logic: Checks if at least two friends are sure about the solution for the current problem.
     * Pre-condition: 'cnt' holds the count of sure friends for the current problem.
     * Invariant: If 'cnt' is 2 or more, 'res' (total implementable problems) is incremented.
     */
    if (cnt >= 2)
      res++;
  }
  // Output the total number of problems that can be implemented.
  cout << res << endl;
}
