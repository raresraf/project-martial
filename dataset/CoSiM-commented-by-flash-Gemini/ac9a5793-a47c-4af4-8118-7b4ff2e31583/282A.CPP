/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - "Bit++".
 * This program simulates operations on a single integer variable 'x',
 * which starts implicitly at 0. Operations are either increment ('++X', 'X++')
 * or decrement ('--X', 'X--'). The goal is to find the final value of 'x' after all operations.
 * Algorithm: Count total increments and decrements, then calculate the net change.
 * Time Complexity: O(T), where T is the number of operations. Each operation involves
 *                  constant time string character access and arithmetic.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 */
#include <iostream> // Required for standard input/output operations (cin, cout).
#include <string>   // Required for std::string.

using namespace std; // Uses the standard namespace.

int main() {
  int t; // Declares an integer to store the total number of operations.
  cin >> t; // Reads the total number of operations from standard input.
  
  string x_str; // Declares a string to store each operation. (Renamed from 'x' to 'x_str' to avoid confusion with variable 'x' in problem statement.)
  int p = 0;    // Counter for increment operations.
  int m = 0;    // Counter for decrement operations.
  
  /**
   * Block Logic: Processes each operation one by one, updating counts of increments and decrements.
   * The loop runs 't' times, decrementing 't' in each iteration.
   * It determines the type of operation by inspecting the second character of the input string.
   * Invariant: `p` holds the total increments, and `m` holds the total decrements, for operations processed so far.
   */
  while (t > 0) { // Loop `t` times, once for each operation.
    cin >> x_str; // Reads the operation string.
    // Checks the second character of the operation string.
    // The problem implies that '++X' and 'X++' always have '+' at x[1],
    // and '--X' and 'X--' always have '-' at x[1].
    if (x_str[1] == '+') { 
      p++; // If '+' is at index 1, it's an increment operation.
    } else {
      m++; // Otherwise, it's a decrement operation.
    }
    t--; // Decrement the loop counter.
  }
  
  // Prints the final value of 'x' (implicitly starting at 0) by subtracting total decrements from total increments.
  cout << p - m;
  
  return 0; // Indicates successful program execution.
}
