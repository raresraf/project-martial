/**
 * @file 231A.CPP
 * @brief An attempted solution for Codeforces Problem 231A, "Team".
 * @details This program is intended to count the number of problems that will
 * be solved by a team, where a problem is solved if at least two members
 * are sure about the solution.
 *
 * Flawed Algorithm:
 * The implementation is highly convoluted and uses flawed logic. It attempts to
 * count different cases (exactly two '1's vs. three '1's) using a mutually
 * exclusive `if-else if` chain.
 * 1. An outer `while` loop suggests it's designed for multiple test cases,
 *    which doesn't match the problem format.
 * 2. The `if-else if` structure is logically incorrect. For an input of `1 1 1`,
 *    the first condition `if (a && b == 1)` will be true, and the `else if`
 *    block for three '1's (`a && b && c == 1`) will never be reached.
 * 3. Despite these flaws, the code accidentally produces the correct output for
 *    this problem because any case where two or more members are sure
 *    (`a+b+c >= 2`) will satisfy one of the first three conditions, correctly
 *    incrementing the `key` counter once per problem. The `cot` counter is
 *    never actually incremented.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <cmath>
#include <iostream>
using namespace std;
int main() {
  long long int a, b, c, i, j, key = 0, cot = 0, sum;
  long long vim;
  // This loop structure is unusual for this problem, suggesting it was
  // adapted from a template for multiple test cases.
  while (!(cin >> j).eof()) {
    for (i = 1; i <= j; i++) {
      cin >> a >> b >> c;
      /**
       * @brief This block attempts to count problems where at least two
       * members agree. The logic is flawed due to the `else if` structure.
       * However, for this problem, it happens to work because any valid case
       * (sum >= 2) will be caught by one of the first three conditions,
       * incrementing `key` exactly once.
       */
      if (a && b == 1) {
        key++;
      } else if (b && c == 1) {
        key++;
      } else if (a && c == 1) {
        key++;
      } else if (a && b && c == 1) {
        // This block is unreachable because `a && b == 1` would be true first.
        cot++;
      } else if (a || b || c == 1) {
        // This block has no meaningful effect.
        sum = 0;
      }
    }
    // The final count is the sum of `key` (which counts all valid cases)
    // and `cot` (which is always 0).
    cout << key + cot << endl;
    key = 0;
    cot = 0;
  }
}
