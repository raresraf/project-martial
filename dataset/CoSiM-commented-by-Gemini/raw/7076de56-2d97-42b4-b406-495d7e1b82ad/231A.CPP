/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 *
 * This program solves a problem where a team's decision to implement a
 * problem depends on the certainty of its members. It reads 'n' sets of
 * three boolean-like integers (0 or 1). It then counts how many of these
 * sets (teams) have at least two members who are sure (represented by 1).
 *
 * @algorithm Simple counting and summation.
 * @time_complexity O(N), where N is the number of teams, as it processes
 *                  each team's opinion in a single pass.
 * @space_complexity O(1) as it only stores a few variables for counting.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to count the number of problems to be implemented.
 */
int main() {
  int i, n, a, b, c, ct = 0, ct1;
  // Read the total number of problems/teams.
  cin >> n;
  /**
   * @brief Loop to iterate through each problem/team.
   * For each of the 'n' problems, this loop reads the opinions of the
   * three team members.
   */
  for (i = 1; i <= n; i++) {
    cin >> a >> b >> c;
    ct1 = 0;
    /**
     * @brief Count the number of members who are sure.
     * This block checks the opinion of each member. If a member is
     * sure (input is 1), a temporary counter `ct1` is incremented.
     */
    if (a == 1)
      ct1++;
    if (b == 1)
      ct1++;
    if (c == 1)
      ct1++;
    /**
     * @brief Check if the team will implement the problem.
     * If the number of sure members is greater than 1 (i.e., at least 2),
     * the main counter `ct` is incremented.
     */
    if (ct1 > 1)
      ct++;
  }
  // Print the total count of problems to be implemented.
  cout << ct;
  return 0;
}