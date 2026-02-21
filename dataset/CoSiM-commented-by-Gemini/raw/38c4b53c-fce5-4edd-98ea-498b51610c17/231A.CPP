/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (Codeforces 231A).
 *
 * @details This program calculates how many problems a three-person team will
 * implement. The team decides to implement a problem only if at least two of
 * the members are sure about the solution.
 *
 * @algorithm The program reads the total number of problems, `n`. It then loops
 * `n` times. In each iteration, it reads three boolean-like integers (0 or 1)
 * representing the confidence of each team member for that problem. It sums up
 * these three integers. If the sum is 2 or greater, it means at least two members
 * are confident, so a counter for implemented problems is incremented. Finally,
 * the total count is printed.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of problems, as it requires a
 *                  single pass through all the problems.
 * Space Complexity: O(1), using only a few variables to store counts and inputs.
 */
#include <iostream>
#include <string> // string is included but not used.
using namespace std;

int main() {
  int n, total = 0, a, b, c;

  cin >> n;
  /**
   * @brief Block Logic: Iterates through each of the `n` problems.
   * @invariant `total` holds the count of problems that will be implemented from
   * the first `i` problems evaluated.
   */
  for (int i = 0; i < n; i++) {
    cin >> a;
    cin >> b;
    cin >> c;
    /**
     * @brief Block Logic: Checks if at least two team members are sure.
     * The sum `a + b + c` represents the number of confident members.
     */
    if (a + b + c >= 2) {
      total++;
    }
  }
  cout << total;
}
