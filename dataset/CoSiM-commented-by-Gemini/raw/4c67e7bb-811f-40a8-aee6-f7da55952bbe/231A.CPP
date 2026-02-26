/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The problem is to determine how many problems a team of three will solve.
 * The team decides to solve a problem only if at least two of the three members
 * are confident about the solution.
 *
 * @algorithm
 * 1. Read the total number of problems `n`.
 * 2. For each of the `n` problems, read the three members' confidence levels
 *    (represented as 1 for confident, 0 for not confident).
 * 3. Sum the three confidence levels. This sum gives the number of members
 *    who are sure.
 * 4. If the sum is 2 or greater, increment a counter for solvable problems.
 * 5. After checking all problems, print the final count.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of problems, because the program
 * iterates through each problem exactly once.
 * Space Complexity: O(N) as used in the code to store intermediate sums,
 * though it could be O(1) as the intermediate storage is unnecessary.
 */
#include <iostream>
using namespace std;

// This array is used to store the sum of opinions for each problem.
// It is not strictly necessary as the check can be done immediately.
int s[1111];

int main() {
  // n: the total number of problems.
  // a, b, c: confidence levels of the three team members for a single problem.
  // solvable_problems_count: a counter for the number of problems that will be implemented.
  int n, a, b, c, solvable_problems_count = 0;

  cin >> n;

  /**
   * @block
   * @brief Loop through each problem to see if it will be solved.
   */
  for (int i = 0; i < n; ++i) {
    // Read the confidence levels for the current problem.
    cin >> a >> b >> c;

    // Sum the opinions. This sum equals the number of sure members.
    s[i] = a + b + c;

    /**
     * @block
     * @brief Check if the problem has enough support to be implemented.
     * @pre The sum of opinions `s[i]` is calculated.
     * @post The `solvable_problems_count` is incremented if the condition is met.
     */
    if (s[i] > 1) // This is equivalent to checking if the sum is >= 2.
      ++solvable_problems_count;
  }

  // Print the total number of problems the team will solve.
  cout << solvable_problems_count;
}