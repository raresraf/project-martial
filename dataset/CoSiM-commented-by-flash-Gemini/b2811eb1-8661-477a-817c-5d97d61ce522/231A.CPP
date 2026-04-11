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
  // Functional Utility: 'kq' (ket qua, meaning result) counts the number of problems that can be implemented.
  int kq = 0;

  // Block Logic: Reads the total number of problems 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Iterates through each problem to evaluate if it can be implemented.
   * Invariant: At the start of each iteration, 'n' represents the remaining problems to process.
   */
  while (n--) {
    // Functional Utility: 'dem' (dem so, meaning count) counts the number of friends who are sure about the current problem.
    int dem = 0;
    // Functional Utility: 'a' is an array to store the opinions of the three friends for the current problem.
    int a[3];

    /**
     * Block Logic: Reads the opinions of the three friends for the current problem.
     * Invariant: After each iteration, 'a[i]' holds an opinion, and 'dem' is updated if the opinion is '1'.
     */
    for (int i = 0; i < 3; i++) {
      // Block Logic: Reads a friend's opinion.
      cin >> a[i];
      /**
       * Block Logic: Checks if the current friend is sure (opinion is '1').
       * Pre-condition: 'a[i]' contains the current friend's opinion.
       * Invariant: 'dem' accurately reflects the number of sure friends so far for this problem.
       */
      if (a[i] == 1)
        dem++;
    }
    /**
     * Block Logic: Checks if at least two friends are sure about the solution for the current problem.
     * Pre-condition: 'dem' holds the count of sure friends.
     * Invariant: If 'dem' is 2 or more, 'kq' (total implementable problems) is incremented.
     */
    if (dem >= 2)
      kq++;
  }
  // Output the total number of problems that can be implemented.
  cout << kq;
  return 0;
}
