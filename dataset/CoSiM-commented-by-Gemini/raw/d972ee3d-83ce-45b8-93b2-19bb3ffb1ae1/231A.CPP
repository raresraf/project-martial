/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 * @details This program counts the number of problems a team will solve. The rule is that
 * a problem is solved if at least two out of three members are confident in the solution.
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter `sols` to 0 to track the number of solvable problems.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three integer inputs (`a`, `b`, `c`), which represent
 *    the confidence of each of the three team members (1 for sure, 0 for not sure).
 * 5. Check if the sum of the three inputs is 2 or greater. This condition is a
 *    direct check for at least two members being sure.
 * 6. If the condition is true, increment the `sols` counter.
 * 7. After the loop finishes, print the total count of solvable problems.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of problems, due to the single loop.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n; // Read the number of problems.
  int sols = 0; // Counter for the number of problems that will be solved.
  
  /**
   * Block Logic: This loop iterates through each problem to determine if it will be solved.
   */
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c; // Read the confidence levels of the three members.
    
    /**
     * Block Logic: If the sum of confidences is 2 or more, it means at least two
     * members agree on the solution, so the team will solve the problem.
     */
    if (a + b + c >= 2)
      sols++;
  }
  
  // Print the final count of solved problems.
  cout << sols;
}