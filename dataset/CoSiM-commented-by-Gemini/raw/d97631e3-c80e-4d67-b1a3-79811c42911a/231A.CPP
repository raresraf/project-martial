/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 * @details The program calculates the number of problems a three-person team will implement.
 * A problem is implemented if at least two of the three members are sure about the solution.
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. Read the total number of problems to evaluate, `n`.
 * 2. Initialize a counter `ans` to 0, which will store the final count of problems to be implemented.
 * 3. Loop `n` times. In each iteration, the loop processes one problem.
 * 4. Read the three boolean values (`p`, `v`, `t`) representing the confidence of each team member.
 * 5. Sum the three values. If the sum is 2 or greater, it means at least two members are confident.
 * 6. If the condition is met, increment the `ans` counter.
 * 7. After the loop completes, print the total count stored in `ans`.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of problems, as it iterates through each problem once.
 * Space Complexity: O(1), as it uses a constant amount of memory.
 */
#include <iostream>

using namespace std;

int main() {

  int n;
  cin >> n;
  int p, v, t;
  int ans = 0; // This will store the final count of problems to be solved.

  /**
   * Block Logic: This loop iterates 'n' times, once for each problem.
   */
  while (n--) {

    cin >> p >> v >> t; // Read the three members' opinions for the current problem.
    
    /**
     * Block Logic: Check if at least two members are sure about the solution.
     * If the sum of their opinions (represented by 1s) is 2 or more, the team
     * will solve the problem.
     */
    if (p + v + t >= 2)
      ans++;
  }

  // Print the final number of problems the team will solve.
  cout << ans << endl;
}