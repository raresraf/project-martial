/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem (Codeforces 231A), a simple counting task.
 * @details This program determines how many problems a team of three will solve,
 * given that they only solve a problem if at least two members are sure about the solution.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read the total number of problems, N.
 * 2. Loop N times:
 *    a. Read the three members' confidence levels (1 for sure, 0 for not sure).
 *    b. If the sum of their confidence levels is 2 or greater, increment a counter.
 * 3. Print the final counter value.
 *
 * @section complexity
 * - Time Complexity: O(N), as the program iterates through N problems once.
 * - Space Complexity: O(1), as storage is constant regardless of input size.
 */
#include <iostream>

using namespace std;

int main() {
  int n, cnt = 0;
  cin >> n;

  /**
   * @brief This loop iterates through each of the 'n' problems.
   * @invariant At the beginning of each iteration, 'cnt' holds the count of
   * problems solvable by the team so far.
   */
  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;

    /**
     * @brief Checks the condition for solving a problem.
     * The condition is that at least two out of the three members are sure.
     * Since 'a', 'b', and 'c' are either 0 or 1, their sum represents the
     * number of members who are sure.
     */
    if ((a + b + c) >= 2)
      cnt++;
  }

  // Output the total number of problems the team will solve.
  cout << cnt << endl;

  return 0;
}
