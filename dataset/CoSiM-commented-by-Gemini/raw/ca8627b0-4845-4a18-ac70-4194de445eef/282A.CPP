/**
 * @file 282A.CPP
 * @brief A clean and efficient solution for Codeforces problem 282A, "Bit++".
 * @details This program simulates a series of simple increment and decrement
 * operations on a variable and outputs the final result. It uses a concise
 * method to identify the operation type.
 *
 * @section Algorithm
 * The solution is based on the insight that the second character of the operation
 * string uniquely identifies the operation.
 *
 * 1. The number of operations, `n`, is read from input.
 * 2. A variable `x` is initialized to 0.
 * 3. The program loops `n` times, reading one operation string `s` per iteration.
 * 4. In each loop, it checks the character at index 1 of the string `s`.
 *    - If `s[1]` is '+', it's an increment operation (`++X` or `X++`).
 *    - Otherwise, it must be a decrement operation (`--X` or `X--`).
 * 5. The variable `x` is updated accordingly.
 * 6. After the loop, the final value of `x` is printed.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of operations, as each
 *   operation is processed in constant time inside a single loop.
 * - Space Complexity: O(1), since the memory usage is constant regardless of `n`.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // n: The number of statements to execute.
  // x: The variable to be operated on, initialized to 0.
  int n, x(0);
  cin >> n;

  string s;
  /**
   * @brief This loop reads and processes each of the `n` statements.
   */
  while (n--) {
    cin >> s;
    /**
     * @brief This block performs the core logic by checking the second character
     * of the operation string to determine if it's an increment or decrement.
     */
    if (s[1] == '+') {
      ++x;
    } else {
      --x;
    }
  }

  // Print the final value of x.
  cout << x << endl;
}
