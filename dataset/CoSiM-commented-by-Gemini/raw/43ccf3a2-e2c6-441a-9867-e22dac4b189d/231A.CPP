/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - Team.
 *
 * @algorithm
 * The problem is to count the number of problems a three-person team will solve.
 * The team's rule for solving a problem is that at least two of the three members
 * must be sure about the solution.
 *
 * The algorithm is as follows:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a `counter` for solved problems to 0.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three members' confidence levels (0 for unsure,
 *    1 for sure) into variables `a`, `b`, and `c`.
 * 5. Check if the sum of `a`, `b`, and `c` is greater than 1. Since the inputs
 *    are only 0 or 1, this is an efficient way to check if at least two members
 *    are sure (i.e., if the sum is 2 or 3).
 * 6. If the condition is met, increment the `counter`.
 * 7. After the loop, print the total count of solved problems.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of problems, as the code
 *   iterates through each problem once.
 * - Space Complexity: O(1), as it only uses a few variables.
 */
#include <iostream>
using namespace std;
int main() {
  int n, counter = 0, a, b, c;
  cin >> n;
  /**
   * @brief This loop processes each of the `n` problems.
   */
  while (n--) {
    cin >> a >> b >> c;
    /**
     * @brief Checks if at least two members are sure.
     * Since sure=1 and not_sure=0, the sum will be > 1 (i.e., 2 or 3)
     * if and only if two or three members are sure.
     */
    if (a + b + c > 1) {
      counter++;
    }
  }
  cout << counter << endl;
}
