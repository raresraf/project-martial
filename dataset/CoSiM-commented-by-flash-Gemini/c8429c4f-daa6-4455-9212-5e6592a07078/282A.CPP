/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A: Bit++.
 * @details This program simulates a simple programming language with a single integer variable 'X',
 * initialized to 0. It processes 'n' operations, where each operation is either an increment (`++X` or `X++`)
 * or a decrement (`--X` or `X--`). The program parses these string operations and calculates
 * the final value of 'X'.
 * @algorithm String parsing and arithmetic simulation.
 * Time Complexity: $O(N \cdot L)$ where N is the number of operations and L is the fixed length of operation strings (typically 3).
 * Space Complexity: $O(L)$ to store the operation string.
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A: Bit++.
 * @details This program simulates a simple programming language with a single integer variable 'X',
 * initialized to 0. It processes 'n' operations, where each operation is either an increment (`++X` or `X++`)
 * or a decrement (`--X` or `X--`). The program parses these string operations and calculates
 * the final value of 'X'.
 * @algorithm String parsing and arithmetic simulation.
 * Time Complexity: $O(N \cdot L)$ where N is the number of operations and L is the fixed length of operation strings (typically 3).
 * Space Complexity: $O(L)$ to store the operation string.
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers (e.g., iostream, string).
using namespace std;     // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: orchestrates the Bit++ language simulation.
   */
  int n, // @var n: The number of operations to process.
      x = 0; // @var x: The integer variable 'X' in the Bit++ language, initialized to 0.
  string operation; // @var operation: Stores the current operation string (e.g., "++X", "X--").

  cin >> n; // Reads the number of operations.

  /**
   * @brief Input validation for 'n'.
   * @block_logic Checks if 'n' is within the valid range [1, 150]. If not, the provided code will proceed
   *              but it's good practice to note this constraint.
   */
  if (1 <= n && n <= 150) { // Although the problem constraints are usually for input, this check is present.
    /**
     * @brief Processes each operation and updates the value of 'X'.
     * @block_logic Iterates `n` times, reading and executing one operation in each iteration.
     * @invariant `n` decreases by 1 in each iteration until it becomes 0. `x` is updated according to the operations.
     */
    while (n--) {
      cin >> operation; // Reads the next operation string.

      /**
       * @brief Parses the operation string and modifies 'x'.
       * @details The operations can be of form `++X`, `X++`, `--X`, `X--`. The code checks the first
       *          character and then the subsequent characters to determine the operation type.
       * @block_logic Multiple `if-else if` blocks are used to match the various operation patterns.
       * @invariant `x` is incremented or decremented by 1 based on the parsed operation.
       */
      if (operation[0] == 'X') { // Handles operations like "X++" or "X--".
        if (operation[1] == '+' && operation[2] == '+') { // Checks for "X++".
          x++; // Increment x.
        } else if (operation[1] == '-' && operation[2] == '-') { // Checks for "X--".
          x--; // Decrement x.
        }

      } else if (operation[0] == '+') { // Handles operations like "++X".
        if (operation[1] == '+' && operation[2] == 'X') { // Checks for "++X".
          x++; // Increment x.
        }

      } else if (operation[0] == '-') { // Handles operations like "--X".
        if (operation[1] == '-' && operation[2] == 'X') { // Checks for "--X".
          x--; // Decrement x.
        }
      }
    }
  }
  // Functional Utility: Prints the final value of 'X'.
  cout << x;
  return 0; // Indicates successful execution.
}
