/**
 * @file 282A.CPP
 * @brief Evaluates a series of increment and decrement operations on a variable.
 *
 * This program simulates a simple programming language with one variable, `x`,
 * initialized to 0. It reads a number of statements, each of which is either
 * an increment (`++X`, `X++`) or a decrement (`--X`, `X--`). It then prints the
 * final value of `x`.
 *
 * @b Algorithm: The program reads the number of statements, `n`. It then iterates `n`
 * times. In each iteration, it reads a statement as a string and compares it to the
 * known increment operations. If it's an increment, `x` is incremented; otherwise, it's
 * decremented.
 *
 * @b Time_Complexity: O(N), where N is the number of statements, because it processes
 * each statement once. String comparison takes constant time for short strings.
 * @b Space_Complexity: O(1), as it only stores the input string for one statement at a time
 * and a few integer variables.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n;
  string s;
  // Read the number of statements.
  cin >> n;
  int c = 0; // The variable 'c' acts as 'x' in the problem description.
  /**
   * @brief This loop processes each statement.
   *
   * Pre-condition: `n` is the number of statements to execute. `c` is initialized to 0.
   * Invariant: After each iteration, `c` is updated according to the operation
   * specified by the input string `s`.
   */
  while (n--) {
    cin >> s;
    // Block Logic: Check if the statement is an increment or decrement.
    if (s == "++X" || s == "X++")
      c += 1;
    else
      c -= 1;
  }
  // Output the final value of the variable.
  cout << c << endl;
}
