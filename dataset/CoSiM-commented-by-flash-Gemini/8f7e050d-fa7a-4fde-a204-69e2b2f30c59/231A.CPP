/**
 * @file 231A.CPP
 * @brief Solves a problem-solving contest scenario: count problems where at least two friends agree.
 *
 * This program simulates a scenario where three friends decide whether to solve a problem.
 * For each of 'n' problems, it takes three binary inputs (0 or 1) indicating each friend's
 * decision. A problem is considered solved if at least two friends decide to solve it.
 * The program then outputs the total count of solved problems.
 *
 * Algorithm: Iterative processing of 'n' problem decisions. For each problem, a logical
 *            condition checks if at least two friends have agreed.
 * Time Complexity: O(N), where N is the number of problems, as each problem is processed once.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  /**
   * @brief Main function: Reads problem decisions and counts solved problems.
   * @return 0 if the program executes successfully.
   */
  int n, c = 0; // 'n': number of problems, 'c': counter for solved problems.
  cin >> n;     // Read the total number of problems.

  /**
   * @brief Loop to process each problem's decisions.
   * @pre 'n' is a positive integer representing the number of problems.
   * @post 'c' accumulates the count of problems solved based on friend agreements.
   * @invariant 'n' decrements in each iteration, processing one problem at a time.
   */
  while (n--) {
    int p, v, t; // 'p', 'v', 't': binary decisions of three friends (1 for agree, 0 for disagree).
    cin >> p >> v >> t; // Read friend decisions for the current problem.

    /**
     * @brief Checks if at least two friends agree to solve the current problem.
     * @pre 'p', 'v', 't' contain binary values.
     * @invariant The condition (p==1 && v==1) || (v==1 && t==1) || (t==1 && p==1)
     *            evaluates to true if any two or three friends agree.
     */
    if ((p == 1 && v == 1) || (v == 1 && t == 1) || (t == 1 && p == 1))
      c++; // Increment the solved problems counter.
  }

  cout << c << endl; // Output the total count of solved problems.

  return 0;
}
