/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem.
 *
 * @details This program calculates the number of problems a team of three will solve.
 * The program reads an integer 'n', representing the number of problems. For each problem,
 * it reads three integers (0 or 1), each representing the confidence of a team member.
 * The team will only solve a problem if at least two members are confident (i.e., their input is 1).
 * The program outputs the total number of problems that will be solved.
 *
 * @section
 * Algorithm:
 * 1. Read the total number of problems, n.
 * 2. Initialize a counter for solvable problems, 'ans', to 0.
 * 3. Loop n times:
 *    a. For each problem, initialize a confidence counter to 0.
 *    b. Read the three confidence values.
 *    c. For each value that is 1, increment the confidence counter.
 *    d. If the confidence counter is 2 or greater, increment 'ans'.
 * 4. Print the final value of 'ans'.
 *
 * Time Complexity: O(n), as it processes each of the n problems with a constant number of operations.
 * Space Complexity: O(1), as the memory usage is constant regardless of the input size.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, ans = 0;
  cin >> n;
  /**
   * @brief This loop iterates through each of the 'n' problems.
   * Invariant: At the beginning of each iteration, 'ans' holds the count of problems
   * solvable by the teams processed so far.
   */
  while (n--) {
    int a[3], count = 0;
    /**
     * @brief This loop reads the confidence levels of the three team members for a single problem.
     * Logic: It reads three integers and increments a 'count' for each '1' encountered.
     */
    for (int i = 0; i < 3; i++) {
      cin >> a[i];
      if (a[i] == 1)
        count++;
    }
    /**
     * @brief Checks if the team has enough confidence to solve the problem.
     * Pre-condition: 'count' holds the number of members (out of 3) who are sure.
     * Logic: If two or more members are sure, the total 'ans' is incremented.
     */
    if (count >= 2)
      ans++;
  }
  cout << ans;

  return 0;
}
