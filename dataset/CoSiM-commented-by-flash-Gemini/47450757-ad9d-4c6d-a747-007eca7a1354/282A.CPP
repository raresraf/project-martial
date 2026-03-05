/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem (Codeforces 282A).
 *
 * This program simulates a simple programming language with a single integer
 * variable, initialized to 0. It processes multiple sets of operations, where
 * each operation increments or decrements the variable. The final value of
 * the variable is printed for each set of operations.
 *
 * Algorithm: Direct simulation. For each set of operations, it reads the
 * number of operations, then iteratively reads and applies each operation
 * (increment or decrement) to the variable.
 * Time Complexity: O(Sum of N across all test cases * L), where N is the
 * number of operations in a test case and L is the constant length of an
 * operation string. This simplifies to O(Total Operations).
 * Space Complexity: O(L) to store the operation string (constant space).
 */
#include <stdio.h>
/**
 * @brief Main function to simulate Bit++ operations and calculate the final value of 'n'.
 *
 * Reads sets of operations from standard input until End-Of-File (EOF).
 * For each set, it initializes 'n' to 0, processes 'a' operations (increment or decrement),
 * and prints the resultant value of 'n'.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare integers: 'i' for loop counter, 'j' (unused), 'a' for number of operations in a set,
  // and 'n' for the variable to be modified, initialized to 0.
  int i, j, a, n = 0;
  // Declare a character array 'str' to store each operation string.
  char str[20];
  /**
   * @brief Block Logic: Processes multiple sets of operations until End-Of-File.
   *
   * This outer loop continues as long as `scanf` successfully reads an integer 'a'.
   * Each iteration represents a new set of operations.
   * Invariant: 'n' is reset to 0 at the beginning of each new set of operations.
   */
  while (scanf("%d", &a) != EOF) {
    /**
     * @brief Block Logic: Processes 'a' operations within the current set.
     *
     * Iterates 'a' times, reading each operation string and updating the variable 'n' accordingly.
     * Pre-condition: 'a' specifies the total number of operations for the current set.
     * Invariant: 'n' is updated based on whether the operation is an increment or decrement.
     */
    for (i = 0; i < a; i++) {
      // Read the operation string into 'str'.
      scanf("%s", str);
      // Pre-condition: 'str' contains a valid operation string ("X++", "++X", "X--", or "--X").
      // Block Logic: Checks if the operation is an increment (by looking at the second or third character for '+').
      if (str[1] == '+' || str[2] == '+')
        n++; // Increment 'n'.
      // Block Logic: Checks if the operation is a decrement (by looking at the second or third character for '-').
      else if (str[1] == '-' || str[2] == '-')
        n--; // Decrement 'n'.
    }
    // Print the final value of 'n' for the current set of operations, followed by a newline.
    printf("%d\n", n);
    // Reset 'n' to 0 for the next set of operations.
    n = 0;
  }
  return 0;
}
