/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The problem is to count how many problems a team of three will
 * implement. The rule for implementation is that at least two of the three
 * members must be sure about the solution.
 *
 * @algorithm
 * 1. Read the number of problems, `n`.
 * 2. Initialize a total counter `count` to 0.
 * 3. An outer loop iterates `n` times for each problem.
 * 4. Inside the outer loop, a temporary counter `surety` is initialized to 0
 *    to track the number of confident members for the current problem.
 * 5. An inner loop iterates 3 times, once for each team member.
 *    a. Read the member's confidence level (1 for sure, 0 for not).
 *    b. If the member is sure, increment the `surety` counter.
 * 6. After the inner loop, if `surety` is 2 or more, increment the total `count`.
 * 7. After all problems are processed, print the final `count`.
 *
 * @complexity
 * Time Complexity: O(n), because for each of the `n` problems, a constant
 * number of operations (3) is performed.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  int n, x;
  int count = 0; // The final count of problems to be solved.
  cin >> n;

  /**
   * @block
   * @brief Outer loop to iterate through each problem.
   */
  for (int i = 0; i < n; i++) {
    // 'surety' counts the number of confident members for the current problem.
    int surety = 0;
    
    /**
     * @block
     * @brief Inner loop to read confidence for the three team members.
     */
    for (int i = 0; i < 3; i++) {
      cin >> x;
      if (x == 1)
        surety++;
    }

    // If at least two members are sure, the team will solve the problem.
    if (surety >= 2)
      count++;
  }

  cout << count << endl;
  return 0;
}
