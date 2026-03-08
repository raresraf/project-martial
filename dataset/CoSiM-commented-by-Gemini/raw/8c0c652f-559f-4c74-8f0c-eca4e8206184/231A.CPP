/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem on Codeforces.
 * @author Ikki (from original source code)
 * @date May 31, 2015 (from original source code)
 *
 * @details
 * This program counts the number of problems a three-person team will implement.
 * The team decides to implement a problem if at least two members are confident
 * about the solution. The program iterates through each problem, sums the
 * confidence votes (1 for sure, 0 for not), and increments a counter if the
 * sum is 2 or greater.
 *
 * @section
 * Algorithm: Iterative counting using a summation condition.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  // n: the total number of problems.
  // ans: the accumulator for the final answer (number of solved problems).
  int n, ans = 0;
  cin >> n;

  /**
   * @brief Main loop to iterate through each of the 'n' problems.
   */
  for (int i = 0; i < n; i++) {
    // a, b, c: confidence votes for the current problem.
    int a, b, c;
    cin >> a >> b >> c;

    /**
     * @brief Check if the team will implement the problem.
     *
     * Logic: Summing the three votes (which are either 0 or 1) and checking
     * if the sum is 2 or more is a direct and efficient way to determine if
     * at least two members are confident.
     */
    if (a + b + c >= 2)
      ans++;
  }
  // Output the total count of problems to be implemented.
  cout << ans << endl;
  return 0;
}
