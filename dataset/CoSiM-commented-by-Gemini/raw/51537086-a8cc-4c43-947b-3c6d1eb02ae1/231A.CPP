/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The problem is to count how many problems a team of three will
 * implement. The rule is that they will implement a problem if at least
 * two of the three members are confident about the solution.
 *
 * @algorithm
 * 1. Read the total number of problems, n.
 * 2. Initialize a counter `count` to 0 for the problems to be solved.
 * 3. Loop n times, once for each problem:
 *    a. Initialize a temporary counter `cnt` to 0 for the current problem.
 *    b. Loop 3 times to read the confidence level of each team member (1 for
 *       confident, 0 for not).
 *    c. Increment `cnt` for each confident member.
 *    d. After checking all three members, if `cnt` is 2 or greater, increment
 *       the main problem counter `count`.
 * 4. After the main loop, print the final `count`.
 *
 * @complexity
 * Time Complexity: O(n), as the code iterates through n problems, and for each
 * problem, it performs a constant number of operations (3 reads).
 * Space Complexity: O(1), as only a few variables are used.
 */
#include <iostream>
using namespace std;

int main() {
  int n; // Number of problems to evaluate.
  cin >> n;

  // 'count' will store the final number of problems to be implemented.
  int count = 0;
  
  /**
   * @block
   * @brief Process each problem to determine if the team will solve it.
   */
  while (n--) {
    // 'cnt' is the number of members confident about the current problem.
    int cnt = 0;
    int temp; // Stores the confidence level of a single member.

    /**
     * @block
     * @brief Read the confidence of the three team members.
     */
    for (int i = 0; i < 3; i++) {
      cin >> temp;
      if (temp == 1)
        cnt++;
    }

    // If two or more members are confident, the team solves the problem.
    if (cnt >= 2)
      count++;
  }

  cout << count << endl;

  return 0;
}
