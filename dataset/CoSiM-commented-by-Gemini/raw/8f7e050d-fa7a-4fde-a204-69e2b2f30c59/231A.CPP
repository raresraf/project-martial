/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem (e.g., Codeforces 231A).
 * @details This program calculates the number of problems a team of three will solve.
 * The team decides to solve a problem only if at least two of the three members
 * are confident in the solution.
 *
 * @algorithm Iterative counting with a conditional check.
 * @complexity
 * - Time: O(n), where 'n' is the number of problems, as it processes each
 *   problem once.
 * - Space: O(1), as it uses a fixed amount of memory.
 */
#include <iostream>

using namespace std;

int main() {
  // n: the total number of problems.
  // c: a counter for the number of problems the team will implement.
  int n, c = 0;
  cin >> n;

  /**
   * @block Problem Processing Loop
   * @logic Iterates through each problem to see if the team will solve it.
   */
  while (n--) {
    // p, v, t: represents the confidence of Petya, Vasya, and Tonya respectively
    // (1 for sure, 0 for not sure).
    int p, v, t;
    cin >> p >> v >> t;

    /**
     * @block Implementation Condition
     * @logic Checks if at least two members are sure.
     * @note A simpler, equivalent condition is `if (p + v + t >= 2)`.
     */
    if ((p == 1 && v == 1) || (v == 1 && t == 1) || (t == 1 && p == 1))
      c++;
  }

  // Output the total count of problems to be solved.
  cout << c << endl;

  return 0;
}
