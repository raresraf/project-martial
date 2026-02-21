/**
 * @file 282A.CPP
 * @brief An elegant solution to the "Bit++" problem (Codeforces 282A).
 *
 * @details This program calculates the final value of a variable 'X' after a
 * series of increment and decrement operations. It uses a particularly clever
 * insight to simplify the logic.
 *
 * @algorithm The program leverages a key observation about the input format:
 * - For increment statements ("X++", "++X"), the character at index 1 is always '+'.
 * - For decrement statements ("X--", "--X"), the character at index 1 is always '-'.
 *
 * By checking only the middle character of the 3-character statement string, the
 * program can determine the correct operation. It iterates `n` times, reading
 * each statement and incrementing or decrementing a counter based on the value
 * of `statement[1]`.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of statements, due to a single pass.
 * Space Complexity: O(1), as it only requires space for a single statement at a time.
 */
#include <cstdio>
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char *argv[]) {
  int n, ans = 0;
  scanf("%d", &n);
  /**
   * @brief Block Logic: Processes `n` statements from the input.
   * @invariant `ans` holds the result of the first `i` operations.
   */
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    // Inspect the middle character to determine the operation.
    if (s.at(1) == '+')
      ans++;
    else
      ans--;
  }
  printf("%d\n", ans);
  return 0;
}
