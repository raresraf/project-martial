/**
 * @file 231A.CPP
 * @brief An unconventional solution to the "Team" competitive programming problem.
 * @details The program determines how many problems a team of three will implement,
 * based on the rule that at least two members must be sure about the solution.
 *
 * @section algorithm Algorithm
 * This implementation uses a single loop to process all inputs, which is an
 * unusual and convoluted approach for this problem.
 * 1. Read the total number of problems, `n`.
 * 2. A single loop iterates from 1 to `3 * n`.
 * 3. Inside the loop, it uses the condition `i % 3 == 0` to distinguish the third
 *    opinion of a problem from the first two.
 * 4. A counter `ni` accumulates the number of '1's (sure opinions) for the current
 *    group of three.
 * 5. Input reading (`cin >> x`) happens inside both branches of the `if/else` statement.
 * 6. After the third opinion is read, the code checks if `ni >= 2`. If so, the
 *    final answer `ans` is incremented.
 * 7. The `ni` counter is reset to 0 after every third opinion is processed, ready for the next problem.
 * 8. Finally, the total count `ans` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of problems. The loop runs 3*N times, which is a linear function of N.
 * - Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <iostream>
using namespace std;
int main() {
  // n: total number of problems.
  // ni: counter for 'sure' opinions (value of 1) for the current problem.
  // ans: final count of problems the team will implement.
  // i: loop counter.
  // x: stores the individual opinion value (0 or 1).
  int n(0), ni(0), ans(0), i(0), x(0);
  cin >> n;

  // This single loop iterates 3*n times to read all individual opinions sequentially.
  for (i = 1; i <= n * 3; i++) {
    // The `i % 3 == 0` condition separates the logic for the third opinion of each problem.
    if (i % 3 == 0 && i > 0) {
      // This block executes for the 3rd, 6th, 9th, etc., opinion.
      cin >> x;
      if (x == 1)
        ni++;
      
      // After processing the third opinion, check if the total 'sure' count is at least 2.
      if (ni >= 2) {
        ans++;
        ni = 0; // Reset counter for the next problem.
      } else
        ni = 0; // Also reset counter if the condition is not met.
    } else {
      // This block executes for the 1st, 2nd, 4th, 5th, etc., opinions.
      cin >> x;
      if (x == 1)
        ni++;
    }
  }
  // Print the total number of problems to be implemented.
  cout << ans << endl;
  return 0;
}
