#include <iostream>
#include <string>
using namespace std;

/**
 * @file 282A.CPP
 * @brief An optimized solution to the "Bit++" problem from Codeforces (Problem 282A).
 * @details This program simulates a series of increment/decrement operations on a variable `X`
 * and prints its final value. This version uses an optimized check.
 *
 * Algorithm:
 * The program initializes a counter `x` to 0. It reads the number of operations `n`.
 * In a loop that runs `n` times, it reads each operation as a string. Instead of
 * comparing the whole string, it astutely checks only the character at index 1.
 * For all possible operations ("++X", "X++", "--X", "X--"), the character at index 1
 * is unique to the type of operation: '+' for increment and '-' for decrement.
 * Based on this character, the counter `x` is updated. Finally, the total value of `x`
 * is printed.
 *
 * Time Complexity: O(n), where `n` is the number of operations. Each operation is
 * processed in constant time.
 * Space Complexity: O(1), as memory usage is constant (the string `str` is reused).
 */
int main() {
  int n;
  cin >> n;
  string str;
  // 'x' holds the value of the variable, initialized to 0.
  int x = 0;
  // Loop through each operation.
  while (n--) {
    cin >> str;
    /**
     * @brief Optimized check for the operation type.
     * The character at index 1 is sufficient to determine whether to
     * increment or decrement.
     */
    if (str[1] == '+') {
      x++;
    } else if (str[1] == '-') {
      x--;
    }
  }
  // Print the final value.
  cout << x;
  return 0;
}
