/**
 * @file 231A.CPP
 * @brief Solution for a programming contest problem known as "Team".
 * @details The problem is to count the number of problems a three-person team will
 * solve. The team decides to solve a problem if at least two of its members are
 * confident about the solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, 'n'.
 * 2. Initialize a counter for solved problems to 0.
 * 3. Loop 'n' times, once for each problem.
 * 4. In each iteration, read the confidence levels of the three team members
 *    (represented as 1 for confident, 0 for not confident).
 * 5. Sum the three confidence levels. If the sum is 2 or greater, it means at
 *    least two members are confident.
 * 6. If the condition is met, increment the solved-problem counter.
 * 7. After processing all problems, print the final count.
 *
 * Time Complexity: O(n), where 'n' is the number of problems.
 * Space Complexity: O(1).
 */
// https://codeforces.com/problemset/problem/231/A
// 231A Team

#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, count = 0;
  cin >> n;
  
  /**
   * @block
   * @description This loop iterates through each of the 'n' problems.
   */
  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;
    
    /**
     * @block
     * @description Checks if the team will solve the problem. The condition
     * `a + b + c >= 2` is a concise way to check if at least two members
     * are confident (represented by a value of 1).
     */
    if (a + b + c >= 2)
      count++;
  }
  
  // Output the total number of problems the team will solve.
  cout << count;
}