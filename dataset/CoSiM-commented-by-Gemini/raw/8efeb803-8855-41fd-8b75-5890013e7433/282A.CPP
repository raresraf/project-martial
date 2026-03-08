/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem (e.g., Codeforces 282A).
 * @details This program simulates a series of increment and decrement operations on a
 * single variable, `x`, initialized to 0. It reads `n` statements and updates `x`
 * based on whether the statement contains "++" or "--".
 *
 * @algorithm Iterative counting based on string parsing.
 * @complexity
 * - Time: O(n), as it iterates `n` times, and the string operations within the loop
 *   are on fixed-size strings, making them effectively O(1).
 * - Space: O(1), as it uses a fixed amount of memory regardless of `n`.
 */
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  // n: the number of statements to execute.
  // x: the variable, initialized to 0.
  int n, x = 0;
  cin >> n;

  string statement, lowSubstr, highSubstr;
  /**
   * @block Statement Processing Loop
   * @logic Iterates through each statement, determines the operation, and updates x.
   */
  for (int i = 0; i != n; ++i) {
    cin >> statement;
    
    // NOTE: This implementation uses substrings to find the operator. A simpler
    // and more robust method would be to check the middle character, e.g.,
    // `if (statement[1] == '+') { x++; } else { x--; }`.

    // Check for "++" as a prefix (e.g., "++X")
    lowSubstr = statement.substr(0, 2);
    // Check for "++" as a suffix (e.g., "X++")
    highSubstr = statement.substr(1, 2); // Corrected from substr(1, 3) which is out of bounds

    if (lowSubstr == "++" || highSubstr == "++") {
      x++;
    } else if (lowSubstr == "--" || highSubstr == "--") {
      x--;
    } 
    // The `else { continue; }` is redundant and has been removed for clarity.
  }
  
  // Output the final value of x.
  cout << x;
  return 0;
}
