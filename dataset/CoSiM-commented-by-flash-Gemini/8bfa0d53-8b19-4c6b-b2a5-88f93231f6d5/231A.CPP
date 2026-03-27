/**
 * @file 231A.CPP
 * @brief Solves a problem where 'a' problems are presented, and for each problem,
 *        three participants indicate their readiness (1 for ready, 0 for not ready).
 *        The goal is to count how many problems will be implemented (i.e., at least two
 *        participants are ready).
 *
 * Algorithm:
 * 1. Read the total number of problems, 'a'.
 * 2. Initialize a counter 'sum' to 0.
 * 3. Loop 'a' times:
 *    a. Read the readiness status (b, c, d) for the three participants for the current problem.
 *    b. If the sum of their readiness statuses (b + c + d) is greater than 1 (meaning 2 or 3), increment 'sum'.
 * 4. Print the final count 'sum'.
 *
 * Time Complexity: O(A) where A is the number of problems, as the program iterates through each problem once.
 * Space Complexity: O(1) as only a few variables are used to store problem-specific data.
 */

#include <iostream>
using namespace std;
// Block Logic: Global declarations for input variables (a, b, c, d) and the sum counter.
// Functional Utility: These variables store problem count, participant readiness, and the result.
// Note: While functional, global variable declarations are generally discouraged in modern C++
// for reasons like increased coupling and potential for unintended side effects.
int a, b, c, d, sum = 0;
int main() {
  // Block Logic: Reads the total number of problems 'a' from standard input.
  cin >> a;
  /**
   * Block Logic: Iterates 'a' times, once for each problem, to read participant
   * readiness and determine if the problem is solvable.
   *
   * Invariant: At the start of each iteration, 'i' represents the current problem index
   * (0 to a-1), and 'sum' holds the count of solvable problems found so far.
   */
  for (int i = 0; i < a; i++) {
    // Block Logic: Reads the readiness status for three participants (b, c, d) for the current problem.
    cin >> b >> c >> d;
    /**
     * Block Logic: Checks if at least two participants are ready for the current problem.
     * Each participant's readiness (b, c, d) is either 0 or 1. Their sum represents
     * the total number of ready participants.
     *
     * Pre-condition: b, c, d are binary (0 or 1).
     * Invariant: If the sum is greater than 1 (i.e., 2 or 3), the problem is considered solvable, and 'sum' is incremented.
     */
    if ((b + c + d) > 1) {
      sum++;
    }
  }
  // Block Logic: Prints the total count of problems that are deemed solvable.
  cout << sum;
}