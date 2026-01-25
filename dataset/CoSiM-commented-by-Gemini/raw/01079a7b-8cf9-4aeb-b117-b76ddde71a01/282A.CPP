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
 * times. In each iteration, it reads a statement as a string and inspects the
 * second character. If the character is `+`, it increments `x`; otherwise, it
 * decrements `x`. This logic correctly handles all four possible statement forms.
 *
 * @b Time_Complexity: O(N), where N is the number of statements, because it processes
 * each statement once.
 * @b Space_Complexity: O(1), as it only stores the input string for one statement at a time
 * and a few integer variables.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n(0), x(0);
  // Read the number of statements.
  cin >> n;
  string s;
  /**
   * @brief This loop processes each statement.
   *
   * Pre-condition: `n` is the number of statements to execute. `x` is initialized to 0.
   * Invariant: After each iteration, `x` is updated according to the operation
   * specified by the input string `s`.
   */
  for (int i = 0; i < n; i++) {
    cin >> s;
    // Block Logic: The operation type is determined by the second character of the statement.
    if (s[1] == '+')
      x++;
    else
      x--;
  }
  // Output the final value of x.
  cout << x << endl;
  return 0;
}