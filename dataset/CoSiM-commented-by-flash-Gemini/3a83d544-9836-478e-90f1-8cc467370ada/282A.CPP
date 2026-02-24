/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - Bit++.
 *
 * This program simulates a simple counter operation. It reads a series of statements,
 * each either incrementing or decrementing a single variable 'X' (represented as `ans`).
 * The goal is to determine the final value of 'X' after all operations.
 *
 * Algorithm: Direct conditional update. The program reads each statement as a string
 * and checks the character at index 1 ('s.at(1)') for '+' or '-' to decide
 * whether to increment or decrement the counter.
 * Time Complexity: O(N * L), where N is the number of statements and L is the
 *                  length of a statement (constant, typically 3).
 * Space Complexity: O(L) for storing a single statement.
 */

#include <cstdio>   // Required for scanf and printf.
#include <iostream> // Required for cin.
#include <string>   // Required for std::string.
using namespace std; // Uses the standard namespace for string, cin, cout.

int main(int argc, char *argv[]) {
  /**
   * @brief Main function to execute the Bit++ problem logic.
   *
   * Reads an integer 'n' representing the number of statements.
   * It then iterates 'n' times, reading each statement. Based on the
   * character at index 1 in the statement, it increments or decrements
   * a counter 'ans'. Finally, it prints the accumulated 'ans'.
   */
  // Declare 'n' to store the number of statements, and 'ans' to store the counter, initialized to 0.
  int n, ans = 0;
  // Read the total number of statements using scanf.
  scanf("%d", &n);
  // Block Logic: Process each statement.
  // Invariant: 'i' iterates from 0 to n-1, processing one statement per loop.
  for (int i = 0; i < n; i++) {
    // Declare a string variable to store the current statement.
    string s;
    // Read the statement using cin.
    cin >> s;
    // Conditional Logic: Check the character at index 1 of the statement.
    // The problem guarantees the format is "++X", "--X", "X++", or "X--".
    // So, the character at index 1 will always be '+' or '-'.
    if (s.at(1) == '+')
      ans++; // If it's '+', increment the counter.
    else
      ans--; // If it's '-', decrement the counter.
  }
  // Print the final accumulated value of 'ans', followed by a newline.
  printf("%d\n", ans);
  return 0; // Indicate successful program execution.
}
