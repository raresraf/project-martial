/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (Codeforces 231A).
 *
 * Problem Statement:
 * A team of three is deciding on which problems to solve. They will solve a
 * problem only if at least two of them are confident about the solution. Given 'n'
 * problems, and for each problem the confidence of the three members (1 for sure,
 * 0 for not sure), count the total number of problems they will solve.
 *
 * Algorithm:
 * The program iterates 'n' times, once for each problem. In each iteration, it
 * reads the three confidence values. The core of the logic is the condition
 * `p + v + t >= 2`. Since the confidence values are either 0 or 1, their sum
 * directly represents the number of members who are sure. If this sum is 2 or
 * greater, it means at least two members agree, and a counter for the solved
 * problems is incremented. Finally, the total count is printed.
 *
 * Time Complexity: O(n), as the program performs a single pass through the 'n' problems.
 * Space Complexity: O(1), as it uses a fixed amount of memory regardless of the input size.
 */
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]) {
  int n, p, v, t, ans = 0;
  // Read the total number of problems.
  cin >> n;

  /**
   * Block Logic: Iterate through each problem to determine if it will be solved.
   */
  for (int i = 0; i < n; i++) {
    // Read the confidence levels of the three members.
    cin >> p >> v >> t;
    // Check if at least two members are sure.
    if (p + v + t >= 2)
      ans++; // Increment the count of solvable problems.
  }
  
  // Print the final count.
  cout << ans;
  return 0;
}
