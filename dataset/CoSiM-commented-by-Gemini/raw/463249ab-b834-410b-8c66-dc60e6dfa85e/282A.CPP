/**
 * @file raw/463249ab-b834-410b-8c66-dc60e6dfa85e/282A.CPP
 * @brief Solution to Codeforces problem 282A - Bit++.
 *
 * @algorithm
 * The problem involves a variable 'x', initially 0, which is modified by a
 * series of increment or decrement statements. The statements can be either
 * prefix or postfix (e.g., "++X", "X++", "--X", "X--"). The goal is to compute
 * the final value of 'x'.
 *
 * This solution directly simulates the process described in the problem statement.
 *
 * The algorithm is as follows:
 * 1. Read `n`, the number of statements to be executed.
 * 2. Initialize an integer variable `x` to 0.
 * 3. Loop `n` times to process each statement.
 * 4. In each iteration, read the statement as a string.
 * 5. Use `if-else if` conditions to check the content of the string:
 *    - If the string is "X++" or "++X", it's an increment operation, so `x` is incremented.
 *    - If the string is "X--" or "--X", it's a decrement operation, so `x` is decremented.
 * 6. After the loop completes, print the final value of `x`.
 *
 * This approach is clear, readable, and correctly solves the problem by
 * handling all specified operation types.
 *
 * @complexity
 * - Time Complexity: O(n), where `n` is the number of statements, because the
 *   loop runs `n` times and the string comparisons inside are constant time.
 * - Space Complexity: O(1), as it uses a few variables of constant size. The
 *   string `s` is reused in each iteration.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;    // To store the operation string (e.g., "++X").
  int n;       // The number of statements.
  int x = 0;   // The variable `x`, initialized to 0.
  
  cin >> n;

  /**
   * @brief Loop through each statement, updating `x` accordingly.
   */
  for (int i = 0; i < n; i++) {
    cin >> s;
    // Check for increment operations.
    if (s == "X++" || s == "++X") {
      x++;
    } 
    // Check for decrement operations.
    else if (s == "X--" || s == "--X") {
      x--;
    }
  }

  // Print the final value of x.
  cout << x << endl;
  
  return 0;
}
