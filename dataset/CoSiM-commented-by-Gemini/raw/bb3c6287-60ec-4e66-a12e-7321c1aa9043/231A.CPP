/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem from Codeforces.
 *
 * @details This program calculates the number of problems a three-person team will solve.
 * The team decides to solve a problem only if at least two members are sure about the solution.
 * The input for each problem is three integers (0 or 1), representing the confidence of each member.
 *
 * @section
 * Algorithm:
 * 1. Read the total number of problems, n.
 * 2. Initialize a counter for solvable problems to 0.
 * 3. Loop n times for each problem:
 *    a. Read the three confidence values (x, y, z).
 *    b. Since the values are 0 or 1, their sum directly equals the number of confident members.
 *    c. If the sum is 2 or greater, increment the counter.
 * 4. Print the total count of solvable problems.
 *
 * Time Complexity: O(n), as it makes a single pass through the n problems.
 * Space Complexity: O(1), as it uses a constant amount of memory.
 */

/*
    Problem Link    :   https://codeforces.com/problemset/problem/231/A
    Contest Link    :   https://codeforces.com/contest/231/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own.
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int count = 0;
  /**
   * @brief This loop iterates through each of the 'n' problems.
   * Invariant: 'count' holds the number of problems solvable by teams processed so far.
   */
  while (n--) {
    int x, y, z;
    cin >> x >> y >> z;
    /**
     * @brief Checks if the team has enough confidence to solve the problem.
     * Logic: The sum of the three inputs (0 or 1) equals the number of confident members.
     * If this sum is 2 or more, the problem is considered solvable.
     */
    if ((x + y + z) >= 2)
      count++;
  }
  cout << count << endl;
  return 0;
}