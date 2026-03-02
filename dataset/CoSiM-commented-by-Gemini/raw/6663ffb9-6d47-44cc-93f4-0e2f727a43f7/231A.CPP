// By archit-dev, contest: Codeforces Round #143 (Div. 2), problem: (A) Team,
// Accepted, #, Copy

/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem from Codeforces Round #143 (Div. 2).
 * @details This program determines the number of problems a three-person team will implement.
 * The team decides to solve a problem if at least two of the three members are confident in their solution.
 *
 * Algorithm:
 * The program reads the total number of problems, `n`. It then iterates `n` times, reading the
 * confidence level of each of the three members (Petya, Vasya, and Tonya) for each problem.
 * The confidence levels are given as integers (1 for sure, 0 for not sure).
 * The sum of their confidence levels is checked. If the sum is 2 or greater, it indicates that
 * at least two members are sure, and a counter for solved problems is incremented.
 *
 * Time Complexity: O(n), as the program iterates through each of the `n` problems once.
 * Space Complexity: O(1), as the memory used is constant and does not depend on the number of problems.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, petya, vatsya, tonya, count = 0;
  // Read the total number of problems.
  cin >> n;
  /**
   * @brief Loop through each problem to determine if it will be solved.
   * Pre-condition: `n` holds the total number of problems.
   * Invariant: After each iteration, the `count` variable holds the number of problems
   * solved so far among the problems processed.
   */
  for (int i = 0; i < n; i++) {
    // Read the confidence levels of the three members for the current problem.
    cin >> petya >> vatsya >> tonya;
    /**
     * @brief Check if the team will solve the current problem.
     * The condition `petya + vatsya + tonya >= 2` is true if at least two members
     * are confident about the solution (indicated by a value of 1).
     */
    if (petya + vatsya + tonya >= 2) {
      // Increment the counter for solved problems.
      count++;
    } else
      continue;
  }
  // Output the total number of problems the team will solve.
  cout << count << "\n";
  return 0;
}
