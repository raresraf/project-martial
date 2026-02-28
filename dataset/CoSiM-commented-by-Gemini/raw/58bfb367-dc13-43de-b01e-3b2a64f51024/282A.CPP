/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 * @details This problem requires calculating the final value of a variable 'X' after a series
 * of increment or decrement operations.
 *
 * @algorithm
 * The solution initializes a counter `x` to zero. It reads the number of operations `n`
 * and then loops `n` times. In each iteration, it reads an operation as a string.
 * Instead of comparing the whole string, this implementation manually inspects the
 * characters at specific indices (0, 1, and 2) to determine the type of operation.
 * It checks for patterns like "X++", "++X", "X--", and "--X" by verifying the positions
 * of the operators ('+' or '-') and the alphabetical character ('X'). Based on the
 * pattern found, it increments or decrements the counter `x`. Finally, it prints the
 * final value of `x`.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of operations. The loop runs N times, and
 * the operations inside (string reading and character inspection) are effectively constant time
 * for the fixed-length strings of this problem.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <iostream>
#include <string>
using namespace std;

// Type aliases for integer types.
typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  /**
   * @variable n: The total number of operations.
   */
  lli n;
  cin >> n;
  /**
   * @variable x: The counter, initialized to 0.
   */
  lli x = 0;
  /**
   * @block
   * @brief This loop processes each of the `n` given operations.
   * @invariant `x` holds the value accumulated from all previously processed operations.
   */
  while (n--) {
    /**
     * @variable a: The string representing the current operation (e.g., "++X").
     */
    string a;
    cin >> a;
    
    /**
     * @block
     * @brief This block identifies the operation type by inspecting the string's characters.
     */
    // Pre-condition: Check for post-increment ("X++").
    if (isalpha(a[0]) && (a[1]) == '+' && a[2] == '+') {
      x++;
    // Pre-condition: Check for pre-increment ("++X").
    } else if (isalpha(a[2]) && (a[0] == '+') && (a[1] == '+')) {
      x++;
    // Pre-condition: Check for pre-decrement ("--X").
    } else if (isalpha(a[2]) && (a[0] == '-') && (a[1] == '-'))
      x--;
    // Pre-condition: Check for post-decrement ("X--").
    else if (isalpha(a[0]) && (a[1] == '-') && (a[2] == '-'))
      x--;
  }
  // Output the final value of the counter.
  cout << x << endl;
  return 0;
}