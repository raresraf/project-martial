/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - Bit++.
 *
 * This program simulates a simple counter operation. It reads a series of statements,
 * each either incrementing or decrementing a single variable 'X' (represented as `value`).
 * The goal is to determine the final value of 'X' after all operations.
 *
 * Algorithm: Direct conditional update. The program reads each statement into
 * a character array and checks for '+' or '-' at specific positions to decide
 * whether to increment or decrement the counter.
 * Time Complexity: O(N * L), where N is the number of statements and L is the
 *                  length of a statement (constant, typically 3 or 4).
 * Space Complexity: O(L) for storing a single statement.
 */

#include <iostream> // Required for standard input/output operations (cin, cout).

using std::cin;  // Use std::cin without the std:: prefix.
using std::cout; // Use std::cout without the std:: prefix.

int main(void) {
  /**
   * @brief Main function to execute the Bit++ problem logic.
   *
   * Reads an integer 'n' representing the number of statements.
   * It then iterates 'n' times, reading each statement. Based on the
   * characters in the statement, it increments or decrements a counter
   * 'value'. Finally, it prints the accumulated 'value'.
   */
  // Declare 'n' to store the number of statements.
  int n;
  // Read the total number of statements.
  cin >> n;

  // Initialize 'count' for the loop iteration, and 'value' as the counter.
  int count = 0;
  int value = 0;
  // Block Logic: Process each statement.
  // Invariant: 'count' increments from 0 to n-1, processing one statement per iteration.
  while (count++ < n) {
    // Declare a character array to store the statement string.
    // Initialized to all zeros for safety, ensuring null termination.
    char statement[] = {0, 0, 0, 0};
    // Read the statement into the character array.
    cin >> statement;
    // Conditional Logic: Check for '+' or '-' to update 'value'.
    // If '+' is found at the first or third character (0-indexed).
    if (statement[0] == '+' || statement[2] == '+') {
      value++; // Increment the counter.
    }
    // If '-' is found at the first or third character.
    else if (statement[0] == '-' || statement[2] == '-') {
      value--; // Decrement the counter.
    }
  }
  // Output the final accumulated value.
  cout << value;

  return 0; // Indicate successful program execution.
}