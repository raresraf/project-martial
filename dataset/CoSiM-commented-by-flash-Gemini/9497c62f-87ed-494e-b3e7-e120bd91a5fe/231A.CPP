/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (Problem 231A).
 * @details This program determines how many problems a team of three will implement.
 * A problem is implemented if at least two of the three members are confident
 * in their solution.
 *
 * @section Algorithm
 * The program first reads an integer `t`, representing the total number of problems.
 * It then enters a loop that iterates `t` times. In each iteration, it does the following:
 * 1. Reads the three integer votes (`a`, `b`, `c`), which are either 0 or 1.
 * 2. Initializes a temporary counter `d` to 0.
 * 3. Individually checks if each vote is 1, incrementing `d` for each confident member.
 * 4. After counting the votes for the current problem, it checks if `d` is 2 or greater.
 * 5. If the condition is met, a primary counter `e` is incremented.
 * After the loop completes, the final value of `e` is printed, representing the
 * total number of problems that will be implemented.
 *
 * @section Complexity
 * - Time Complexity: O(t), where 't' is the number of problems, as the program
 *   iterates through each problem once.
 * - Space Complexity: O(1), as memory usage is constant.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @var t The total number of problems to evaluate.
   * @var a, b, c The confidence votes (0 or 1) of the three team members.
   * @var d A temporary counter for votes on a single problem.
   * @var e The final count of problems that will be implemented.
   */
  int t, a, b, c, d, e = 0;

  // Read the total number of problems.
  cin >> t;

  /**
   * @brief Main loop to process each of the 't' problems.
   */
  while (t--) {
    // Read the three votes for the current problem.
    cin >> a;
    cin >> b;
    cin >> c;

    // Reset the temporary vote counter for the current problem.
    d = 0;

    /**
     * @brief Block Logic: Manually counts the number of confident members (votes of '1').
     */
    if (a == 1) {
      d = d + 1;
    }
    if (b == 1) {
      d = d + 1;
    }
    if (c == 1) {
      d = d + 1;
    }

    /**
     * @brief Logic: Checks if at least two team members are confident.
     */
    if (d >= 2) {
      // If so, increment the total count of problems to be implemented.
      e = e + 1;
    }
  }
  
  // Output the final count.
  cout << e;
}
