/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem.
 *
 * @details
 * This program determines how many problems a team of three will decide to
 * implement. The rule is that a problem is implemented if at least two of the
 * three members are confident in the solution. The program reads the number
 * of problems and then, for each problem, the confidence level (1 for sure,
 * 0 for not sure) of each member. It counts and outputs the total number of
 * problems that will be implemented.
 *
 * @section
 * Algorithm: Iterative counting based on a summation condition.
 * Time Complexity: O(N), where N is the number of problems to evaluate.
 * Space Complexity: O(1), as variables are reused for each problem.
 */
#include <iostream>

using namespace std;

int main() {
  // n: the total number of problems.
  // a, b, c: confidence levels of the three team members for a single problem (1 or 0).
  int n, a, b, c;
  // counter: accumulates the number of problems the team will implement.
  int counter = 0;
  cin >> n;

  /**
   * @brief This loop iterates through each of the 'n' problems.
   *
   * Invariant: At the start of each iteration, 'counter' holds the total
   * count of implementable problems from the preceding iterations.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    /**
     * @brief Checks if the team will implement the current problem.
     *
     * Logic: The sum of the confidence levels (a + b + c) will be 2 or 3 if
     * at least two members are sure. The condition `a + b + c > 1` is a
     * concise way to check for this.
     */
    if (a + b + c > 1)
      counter++;
  }

  // Output the final count of problems to be implemented.
  cout << counter << endl;
  return 0;
}
