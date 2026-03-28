/**
 * @file 282A.CPP
 * @brief Evaluates a series of increment and decrement operations.
 *
 * This program reads a number of operations and then the operations themselves,
 * which are given as strings like "++X", "X++", "--X", or "X--". It calculates
 * the final value of a variable 'x' that starts at 0 and is modified by these
 * operations.
 *
 * @b Algorithm: The program initializes a counter 'x' to 0. It then loops through
 *    each operation. For each operation string, it checks if it contains a '+'
 *    or a '-' character. If a '+' is found, 'x' is incremented. If a '-' is
 *    found, 'x' is decremented.
 *
 * @b Time_Complexity: O(N), where N is the number of operations, as each
 *    operation is processed in constant time.
 * @b Space_Complexity: O(L), where L is the maximum length of the operation
 *    string (which is constant in this problem, so effectively O(1)).
 */
#include <iostream>
using namespace std;
// Note: The use of `main()` without a return type is non-standard in C++.
// It should be `int main()`.
main() {
  int n;
  cin >> n;
  int x = 0; // The variable to be incremented or decremented.
  
  /**
   * @brief This loop reads and processes each operation.
   */
  while (n > 0) {
    string word;
    cin >> word;
    int len = word.length();
    /**
     * @brief This block checks the operation type.
     * It checks the first or last character of the string for '+' or '-'.
     * This works for both prefix and postfix notations like "++X", "X++", etc.
     */
    if (word[0] == '+' || word[len - 1] == '+')
      x = x + 1;
    else if (word[0] == '-' || word[len - 1] == '-')
      x = x - 1;
    n = n - 1;
  }
  cout << x << endl;
}
