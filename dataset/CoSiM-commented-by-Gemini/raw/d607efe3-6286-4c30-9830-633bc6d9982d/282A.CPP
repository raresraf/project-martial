/**
 * @file 282A.CPP
 * @brief An efficient solution for the "Bit++" problem from Codeforces (282A).
 * @details This program simulates a simple programming language with one variable 'x'
 * and four operations: "++X", "X++", "--X", and "X--". It calculates the final
 * value of 'x' after a series of 'n' operations. This implementation uses a
 * clever shortcut by observing that the second character of the operation string
 * is always '+' for an increment and '-' for a decrement.
 *
 * @algorithm Character-based operation checking.
 * @complexity
 * Time: O(N), where N is the number of operations, as it processes each string once.
 * Space: O(1), as the string is reused in each iteration.
 */
#include <iostream>
#include <string> // Recommended to include for std::string

using namespace std;

int main(int argc, char const *argv[]) {
  int n; // Number of operations
  cin >> n;

  int x = 0; // The variable, initialized to 0

  /**
   * @brief Main loop to process each operation.
   */
  while (n--) {
    string str;
    cin >> str;

    /**
     * @brief Check the second character to determine the operation.
     * For "++X" and "X++", str[1] is '+'.
     * For "--X" and "X--", str[1] is '-'.
     */
    if (str[1] == '+') {
      x++;
    } else { // It is guaranteed by the problem that the character will be '-'
      x--;
    }
  }

  // Print the final value of x.
  cout << x;

  return 0; // Indicate successful execution.
}