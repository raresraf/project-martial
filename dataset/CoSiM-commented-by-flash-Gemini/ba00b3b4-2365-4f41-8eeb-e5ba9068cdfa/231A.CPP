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
  // Functional Utility: 'n' stores the total number of problems to evaluate.
  int n;
  // Functional Utility: 'ans' accumulates the total count of implementable problems.
  int ans = 0;

  // Block Logic: Reads the total number of problems 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Iterates through each problem to evaluate if it can be implemented.
   * Invariant: At the start of each iteration, 'n' represents the remaining problems to process.
   */
  while (n--) {
    // Functional Utility: 'a' is an array to store the opinions of the three friends for the current problem.
    int a[3];
    // Functional Utility: 'count' counts the number of friends who are sure about the current problem.
    int count = 0;

    /**
     * Block Logic: Reads the opinions of the three friends for the current problem and counts sure friends.
     * Invariant: After each iteration, 'a[i]' holds an opinion, and 'count' is updated if the opinion is '1'.
     */
    for (int i = 0; i < 3; i++) {
      // Block Logic: Reads a friend's opinion.
      cin >> a[i];
      /**
       * Block Logic: Checks if the current friend is sure (opinion is '1').
       * Pre-condition: 'a[i]' contains the current friend's opinion.
       * Invariant: 'count' accurately reflects the number of sure friends so far for this problem.
       */
      if (a[i] == 1)
        count++;
    }
    /**
     * Block Logic: Checks if at least two friends are sure about the solution for the current problem.
     * Pre-condition: 'count' holds the number of sure friends.
     * Invariant: If 'count' is 2 or more, 'ans' (total implementable problems) is incremented.
     */
    if (count >= 2)
      ans++;
  }
  // Output the total number of problems that can be implemented.
  cout << ans;

  return 0;
}
