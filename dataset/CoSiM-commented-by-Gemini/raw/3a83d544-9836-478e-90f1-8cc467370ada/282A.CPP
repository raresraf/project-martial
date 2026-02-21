/**
 * @file 282A.CPP
 * @brief Calculates the final value of a variable after a series of increment
 *        and decrement operations.
 *
 * This program simulates a programming language where a variable `X` starts at 0
 * and can be modified by statements like "++X", "X++", "--X", and "X--". It
 * reads a number of such statements and computes the final value of `X`.
 *
 * Algorithm:
 * The program uses a clever simplification. For any of the valid statements,
 * the character at index 1 is unique to the operation type.
 * - For "++X" and "X++", the character at index 1 is '+'.
 * - For "--X" and "X--", the character at index 1 is '-'.
 *
 * 1. Read an integer `n`, the total number of statements.
 * 2. Initialize a counter `ans` to 0.
 * 3. Loop `n` times:
 *    a. Read the statement string.
 *    b. Check the character at index 1.
 *    c. If it is '+', increment `ans`.
 *    d. Otherwise, decrement `ans`.
 * 4. After the loop, print the final value of `ans`.
 *
 * Time Complexity: O(N), where N is the number of statements, as it performs
 * a single pass, and each operation within the loop is O(1).
 * Space Complexity: O(1), as the space used does not scale with the input size.
 */
#include <cstdio>
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char *argv[]) {
  int n, ans = 0;
  scanf("%d", &n);
  
  /**
   * @brief Loop through each statement to update the answer.
   * Pre-condition: `n` is the number of statements to process.
   * Invariant: `ans` holds the accumulated value from the statements processed so far.
   */
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    
    // Check the character at index 1 to determine the operation.
    if (s.at(1) == '+')
      ans++;
    else
      ans--;
  }
  
  // Print the final result.
  printf("%d\n", ans);
  return 0;
}
