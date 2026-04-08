/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (Problem 231A).
 * @details The program determines how many problems a team will solve. The team decides to
 * solve a problem if at least two of the three members are confident about the solution.
 *
 * @algorithm
 * 1. Read the total number of problems, `t`.
 * 2. Initialize a counter for solved problems to zero.
 * 3. Loop `t` times, once for each problem.
 * 4. In each loop, read the confidence levels (0 or 1) of the three members.
 * 5. If the sum of their confidence levels is 2 or greater, increment the counter.
 * 6. After the loop, print the total count of problems they will solve.
 *
 * @time_complexity O(t), where `t` is the number of problems.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
#define sc scanf
#define t1(t) scanf("%d", &t)
#define FOR(x) for (int i = 0; i < x; i++)
#define pf printf
#define wh(n) while (n--)
#define ll long long
using namespace std;

int main() {
  int t; // Represents the number of problems to be evaluated.
  int count = 0; // Counter for the number of problems the team will solve.
  t1(t);

  /**
   * @block_logic Loop through each problem to see if the team will solve it.
   * @pre `t` contains the total number of problems.
   * @post `count` will hold the number of problems with at least two confident members.
   */
  wh(t) {
    int x, y, z; // Confidence levels of the three team members (1 for sure, 0 for not).
    cin >> x >> y >> z;

    // Condition to check if at least two members are sure.
    // This is equivalent to checking if (x + y + z) >= 2.
    if (x && y || x && z || y && z)
      count++;
  }

  // Print the total number of problems that will be solved.
  pf("%d\n", count);
}
