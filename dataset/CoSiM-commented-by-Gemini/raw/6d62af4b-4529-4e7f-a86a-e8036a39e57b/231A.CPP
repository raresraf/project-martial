
#include <bits/stdc++.h>
using namespace std;

/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A, "Team".
 * @details The program counts the number of problems a three-person team will
 * decide to implement. The rule is that they will only solve a problem if at
 * least two of the three members are confident in the solution.
 *
 * Algorithm:
 * The program reads the total number of problems, `n`. It then loops `n` times.
 * In each iteration, it reads the three members' confidence levels (1 for sure,
 * 0 for not sure). By summing these three values, it gets the total number of
 * members who are sure. If this sum is 2 or greater, it means the team will
 * solve the problem, and a counter is incremented. Finally, the total count
 * of problems to be solved is printed.
 *
 * Time Complexity: O(N), where N is the number of problems, as it processes
 * each problem once.
 * Space Complexity: O(1).
 */
int main() {
  // n: the number of problems to consider.
  int n;
  cin >> n;

  // x, y, z: confidence levels of the three team members (0 or 1).
  // sum: stores the number of members sure about a problem.
  // count: total number of problems the team will implement.
  int x, y, z, sum = 0, count = 0;

  /**
   * @brief Process each of the n problems.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    // The sum of the inputs equals the number of members who are sure.
    sum = x + y + z;

    // The problem is implemented if at least two members are sure.
    if (sum >= 2)
      count++;
  }
  cout << count << endl;
  return 0;
}
