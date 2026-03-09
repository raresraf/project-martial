/*
Author : hardRocK
Date :
Online Judge : Codeforces
Problem Name : Team
Problem Link : http://codeforces.com/problemset/problem/231/A
*/

/**
 * @file 231A.CPP
 * @brief Solution to Codeforces Problem 231A: "Team".
 *        This program reads the number of problems and for each problem,
 *        three binary votes (0 or 1) from participants. It determines
 *        how many problems a team can implement if at least two participants
 *        agree to implement it.
 *
 * Algorithm: Direct conditional checking. For each problem, it checks a boolean
 *            expression to see if at least two out of three votes are '1'.
 *            If the condition is met, a global counter for solvable problems is incremented.
 * Time Complexity: O(N), where N is the number of problems. Each problem involves
 *                  a fixed number of input operations and a constant-time boolean check.
 * Space Complexity: O(1) as it uses a fixed amount of memory regardless of input size.
 */

#include <bits/stdc++.h> // Includes various standard libraries for C++ competitive programming.
using namespace std;

int main() {
  /**
   * Functional Utility: Manages the input and processing for a series of problems
   *                     to determine the total count of solvable problems based on
   *                     team consensus.
   */

  int T, cnt = 0; // T: number of test cases (problems); cnt: total count of solvable problems.
  cin >> T;       // Reads the number of test cases from standard input.

  /**
   * Block Logic: Processes each test case (problem).
   * Pre-condition: `T` holds the number of problems. `cnt` is initialized to 0.
   * Invariant: `T` decrements with each iteration; the loop continues until all
   *            problems are evaluated. `cnt` reflects the number of solvable
   *            problems encountered so far.
   */
  while (T--) {

    int a, b, c; // Variables to store the three participants' votes (0 or 1).
    cin >> a >> b >> c; // Reads the votes for the current problem.

    /**
     * Block Logic: Checks if at least two out of three participants voted '1'.
     * Pre-condition: `a`, `b`, and `c` contain binary votes for the current problem.
     * Invariant: `cnt` is incremented if the condition for solvability is met.
     */
    if ((a == 1 && b == 1) || (a == 1 && c == 1) || (b == 1 && c == 1) ||
        (a == 1 && b == 1 && c == 1)) { // Condition Check: At least two votes are '1'.
      cnt++; // Increment the total count of solvable problems.
    }
  }

  // Output: Prints the final total count of problems that can be solved.
  cout << cnt << endl;
  // system("pause"); // Commented out: typically used in Windows for pausing console output, not standard practice.
  return 0;          // Indicates successful program termination.
}
