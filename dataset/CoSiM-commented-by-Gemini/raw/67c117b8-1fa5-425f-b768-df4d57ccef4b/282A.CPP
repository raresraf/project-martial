// 282A

#include <cstring>
#include <iostream>
using namespace std;

/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem from Codeforces (Problem 282A).
 * @details This program simulates a series of increment and decrement operations on a
 * variable, `X`, and outputs its final value.
 *
 * Algorithm:
 * A variable, `res`, is initialized to 0 to represent the initial value of `X`.
 * The program then reads an integer `n`, the total number of operations.
 * It enters a loop that runs `n` times. In each iteration, it reads an operation
 * as a string. It checks if the string corresponds to an increment operation
 * ("X++" or "++X") or a decrement operation ("X--" or "--X"). The `res`
 * variable is incremented or decremented accordingly. After all operations are
 * processed, the final value of `res` is printed.
 *
 * Time Complexity: O(n), where `n` is the number of operations, as each operation
 * is processed in constant time within a single loop.
 * Space Complexity: O(1), as the memory usage is constant regardless of the
 * number of operations (the string for the operation is reused).
 */
int main() {
  int n;
  string s;
  // `res` holds the final value of X, initialized to 0.
  int res = 0;
  // Read the number of operations.
  cin >> n;
  /**
   * @brief Main loop to process each operation.
   */
  while (n--) {
    cin >> s;
    // Check if the operation is an increment.
    if (s == "X++" || s == "++X")
      res++;
    // Check if the operation is a decrement.
    else if (s == "X--" || s == "--X")
      res--;
  }
  // Print the final value of X.
  cout << res;
  return 0;
}
