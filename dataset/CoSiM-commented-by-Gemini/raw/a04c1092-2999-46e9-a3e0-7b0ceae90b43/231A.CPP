/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (Codeforces 231A).
 * @details This program determines the number of problems a three-person team will
 *          solve. The condition for solving a problem is that at least two of the
 *          three members are confident in the solution.
 *
 * @algorithm The approach is a straightforward counting algorithm. It iterates through
 *            each problem, reads the confidence level of each of the three members
 *            (1 if confident, 0 otherwise), and sums them up. If the total
 *            confidence sum is 2 or greater, it indicates that the team will solve
 *            the problem, and a counter is incremented.
 *
 * @time_complexity O(N), where N is the total number of problems, because each
 *                  problem is processed in constant time.
 * @space_complexity O(1), as the program only uses a few variables to store the
 *                   counts and inputs, regardless of the number of problems.
 */
#include <iostream>

using std::cin;
using std::cout;
using std::endl;

int main() {
  int n;
  cin >> n;
  int a, b, c;
  int problemSolved = 0; // Counter for problems the team will solve.

  /**
   * @brief This loop iterates 'n' times, once for each problem.
   */
  for (int i = 0; i != n; ++i) {
    cin >> a >> b >> c;
    /**
     * @brief A problem is solved if the sum of confidences is at least 2.
     * This checks if at least two team members are sure about the solution.
     */
    if ((a + b + c) >= 2) {
      problemSolved++;
    }
  }
  // Output the final count of solved problems.
  cout << problemSolved;
  return 0;
}
