/**
 * @file 282A.CPP
 * @brief This competitive programming solution processes a sequence of operations
 *        (increment or decrement) on a single variable and outputs its final value.
 *        Operations are typically in the format "X++", "++X", "X--", or "--X".
 * Algorithm: Iterative processing. The program initializes a counter to zero and then
 *            iterates through each operation string. It determines whether the operation
 *            is an increment or decrement by checking a specific character in the string
 *            and updates the counter accordingly.
 * Time Complexity: O(N), where N is the number of operations. Each operation involves
 *                  reading a line and a constant-time character check.
 * Space Complexity: O(1), as only a few fixed-size integer variables and a string for
 *                   the current operation are stored.
 */
#include <cstdio>   // Required for scanf and printf for C-style I/O.
#include <iostream> // Required for cin, cout, and getline for C++ I/O.
#include <string>   // Required for string manipulation.
using namespace std;

/**
 * @brief Main function of the program. Reads operations, processes them, and prints the final value.
 * Functional Utility: Serves as the entry point, orchestrating the processing of variable operations.
 */
int main() {
  int numOperations; // Declares an integer to store the total number of operations.
  // Reads the number of operations using scanf. The "
" consumes the trailing newline character.
  scanf("%d
", &numOperations);
  string temp = ""; // String to temporarily store each operation.
  int output = 0;   // The variable whose value is being modified; initialized to 0.

  // Block Logic: Loops through each operation to update the 'output' variable.
  // Invariant: After each iteration, 'output' reflects the result of all processed operations.
  while (--numOperations >= 0) { // Decrements numOperations and continues as long as it's non-negative.
    getline(cin, temp);          // Reads the entire line (operation string) into 'temp'.
    // Block Logic: Checks the second character of the operation string to determine increment or decrement.
    if (temp[1] == '+') { // If the second character is '+', it's an increment operation.
      ++output;           // Increments the 'output' variable.
    } else {              // Otherwise (the second character must be '-'), it's a decrement operation.
      --output;           // Decrements the 'output' variable.
    }
  }
  printf("%d
", output); // Prints the final calculated value of 'output'.
  return 0;               // Indicates successful program termination.
}
