/**
 * @file 282A.CPP
 * @brief Evaluates a series of "++X", "X++", "--X", "X--" style operations.
 *
 * This program starts with a variable `X` initialized to 0. It then reads a
 * number of statements, each being one of four operations: "++X", "X++", "--X",
 * or "X--". It calculates and prints the final value of `X` after all
 * operations are performed.
 *
 * Algorithm:
 * 1. Read an integer `n`, the number of statements to process.
 * 2. Initialize an integer `value` (representing X) to 0.
 * 3. Loop `n` times to process each statement.
 * 4. In each iteration, read the statement string. The statement will always be
 *    three characters long.
 * 5. Check the characters of the statement to determine the operation. Since the
 *    character 'X' is always in the middle, checking the character at index 0
 *    or 2 is sufficient to determine if it's an increment or decrement.
 * 6. If the statement contains a '+', increment `value`.
 * 7. If the statement contains a '-', decrement `value`.
 * 8. After the loop finishes, print the final `value`.
 *
 * Time Complexity: O(N), where N is the number of statements, as each statement
 * is processed once.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <iostream>
#include <string> // Recommended to include for std::string, though not explicitly used here.

using std::cin;
using std::cout;

int main(void) {
  int n;
  cin >> n;

  int count = 0;
  int value = 0; // This variable represents 'X', starting at 0.
  
  /**
   * @brief Loop through each statement and update the value.
   * Pre-condition: `n` is the number of statements.
   */
  while (count++ < n) {
    char statement[] = {0, 0, 0, 0}; // Buffer to read the statement e.g., "++X".
    cin >> statement;
    
    /**
     * @brief Check for '+' or '-' to determine the operation.
     * For "++X", statement[0] is '+'. For "X++", statement[2] is '+'.
     * The logic is similar for '-'.
     */
    if (statement[0] == '+' || statement[2] == '+') {
      value++;
    } else if (statement[0] == '-' || statement[2] == '-') {
      value--;
    }
  }
  
  // Print the final value of X after all operations.
  cout << value;

  return 0;
}