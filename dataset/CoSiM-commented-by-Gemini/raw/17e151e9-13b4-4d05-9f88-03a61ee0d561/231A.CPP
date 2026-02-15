/**
 * @file 231A.CPP
 * @brief Solution to the "Team" competitive programming problem.
 * @details This program calculates the number of problems a team of three will decide to implement.
 * The condition for implementing a problem is that at least two of the three members must be sure about the solution.
 *
 * @section algorithm Algorithm
 * The algorithm counts the problems that meet the criteria:
 * 1. The outer `while(cin >> n)` loop is a common competitive programming pattern for handling multiple test cases, though this problem likely has only one.
 * 2. For each test case, a counter `cnt` is initialized to zero.
 * 3. An inner loop runs `n` times, once for each problem.
 * 4. In each iteration, it reads the three members' opinions (`p`, `v`, `t`), where 1 means sure and 0 means not sure.
 * 5. It checks if the sum of the opinions is 2 or greater. This is a concise way to determine if at least two members are confident.
 * 6. If the condition is met, the counter `cnt` is incremented.
 * 7. After iterating through all the problems for a test case, the final value of `cnt` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n) for each test case, where `n` is the number of problems, as it iterates through each problem once.
 * - Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, p, v, t, cnt;
  // This outer loop is a common pattern to handle multiple test cases from input.
  while (cin >> n) {
    // 'cnt' will count the number of problems the team decides to implement.
    cnt = 0;
    // Loop through each of the 'n' problems for the current test case.
    while (n--) {
      // Read the confidence level of the three team members (1 for sure, 0 for not sure).
      cin >> p >> v >> t;
      // The original verbose condition checked all combinations for at least two '1's.
      // This is logically equivalent to checking if their sum is 2 or more.
      if (p + v + t >= 2) {
        // If at least two members are sure, the team implements the problem.
        cnt++;
      }
    }
    // Print the total count for the current test case.
    cout << cnt << endl;
  }
  return 0;
}
