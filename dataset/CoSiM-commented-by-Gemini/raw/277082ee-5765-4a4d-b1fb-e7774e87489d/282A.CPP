/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 * @details This program simulates a simple programming language with one variable, X,
 * and two operations: ++X (prefix increment) and X++ (postfix increment), and
 * --X (prefix decrement) and X-- (postfix decrement). It calculates the final
 * value of X after a series of such operations.
 *
 * Algorithm:
 * 1. Initialize a variable 'x' to 0.
 * 2. Read the number of operations, 'n'.
 * 3. Loop 'n' times:
 *    a. Read the operation statement as a string.
 *    b. Check if the string represents an increment operation ("++X" or "X++").
 *       If so, increment 'x'.
 *    c. Otherwise, assume it's a decrement operation and decrement 'x'.
 * 4. Print the final value of 'x'.
 *
 * Time Complexity: O(n), where 'n' is the number of statements, as each
 * statement is processed in constant time.
 * Space Complexity: O(1), as memory usage is constant regardless of input size.
 */
#include <iostream>
#include <string.h>
//#include<ctype.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  int x = 0;
  // Loop through all the given operations.
  while (n--) {
    cin >> s;
    /**
     * Block Logic: Checks the operation type and updates 'x'.
     * It explicitly checks for the two increment variations.
     * Any other operation is implicitly treated as a decrement.
     */
    if (((s[0] == '+') && (s[1] == '+') && (s[2] == 'X')) ||
        ((s[0] == 'X') && (s[1] == '+') && (s[2] == '+'))) {
      x++;
    }
    else {
      x--;
    }
  }
  // Output the final value of x.
  cout << x << endl;

  return 0;
}