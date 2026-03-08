/**
 * @file 231A.CPP
 * @brief An optimal solution for the "Team" problem (e.g., Codeforces 231A).
 * @details This program calculates the number of problems a team will solve, where
 * a problem is solved if at least two out of three members are confident (represented by 1).
 * It uses a very efficient and concise method to check the condition.
 *
 * @algorithm Single-pass iterative counting.
 * @complexity
 * - Time: O(n), where 'n' is the number of problems.
 * - Space: O(1).
 */
#include <iostream>
using namespace std;

// The return type of main defaults to int in standard C++.
main() {
  // n: total number of problems.
  // p, v, t: confidence of Petya, Vasya, and Tonya.
  // ctr: counter for problems to be solved.
  int n, p, v, t, ctr = 0;
  cin >> n;

  /**
   * @block Problem Processing Loop
   * @logic Iterates through each problem, reads the three members' decisions,
   * and increments the counter if the condition is met.
   */
  for (int i = 0; i < n; i++) {
    cin >> p >> v >> t;
    /**
     * @block Implementation Condition
     * @logic This is the most efficient way to check the condition. Since inputs are
     * 0 or 1, their sum equals the number of confident members. If the sum is
     * 2 or 3, the team will solve the problem.
     */
    if (p + v + t >= 2) {
      ctr++;
    }
  }
  
  // Output the final count.
  cout << ctr;
}
