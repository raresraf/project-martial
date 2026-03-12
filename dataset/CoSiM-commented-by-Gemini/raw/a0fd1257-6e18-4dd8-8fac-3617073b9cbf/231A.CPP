/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (Codeforces 231A).
 * @details This program counts the number of problems a three-person team will
 *          decide to implement, based on the condition that at least two members
 *          are confident about the solution.
 *
 * @algorithm The algorithm is a simple counting process. It iterates through each
 *            problem, sums the boolean confidence values (1 for confident, 0 for not)
 *            of the three team members. If this sum is 2 or more, it means the
 *            condition is met, and a counter for solvable problems is incremented.
 *
 * @time_complexity O(N), where N is the number of problems, as each problem is
 *                  evaluated in a single pass.
 * @space_complexity O(1), as only a few variables are needed for the computation,
 *                   irrespective of the input size.
 */
#include <iostream>

using namespace std;

int main() {
  /**
   * @brief Optimizes I/O operations in C++.
   * `ios::sync_with_stdio(0)` unties C++ streams from C streams.
   * `cin.tie(0)` unties cin from cout.
   * This is a standard practice in competitive programming for faster I/O.
   */
  ios::sync_with_stdio(0);
  cin.tie(0);

  int n;
  cin >> n;

  int probs = 0; // Counter for the number of problems to be solved.

  /**
   * @brief Loop through each of the 'n' problems.
   */
  for (int i = 0; i < n; i++) {
    int p, v, t; // Confidence levels of the three members.
    cin >> p >> v >> t;

    /**
     * @brief If the sum of confidence levels is 2 or more, the team solves the problem.
     */
    if (p + v + t >= 2) {
      probs++;
    }
  }

  // Output the total count of solvable problems.
  cout << probs << "\n";

  return 0;
}
