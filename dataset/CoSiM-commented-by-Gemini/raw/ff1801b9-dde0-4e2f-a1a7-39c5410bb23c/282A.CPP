/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem (282A) on Codeforces.
 *
 * @details The program simulates a simple programming language with one variable, X,
 * initialized to 0. It processes a series of operations ("++X", "X++", "--X", "X--")
 * and calculates the final value of X.
 *
 * Algorithm:
 * 1. Read the number of statements to be executed.
 * 2. Initialize a counter 'total' to 0.
 * 3. Loop for the number of statements:
 *    a. Read the entire operation as a string line.
 *    b. A clever shortcut is used: check the character at index 1. For both
 *       "++X" and "X++", the character s[1] is '+'. For "--X" and "X--", it's '-'.
 *    c. If s[1] is '+', increment 'total'. Otherwise, decrement it.
 * 4. Print the final value of 'total'.
 *
 * Time Complexity: O(N), where N is the number of statements.
 * Space Complexity: O(1), as the string is reused for each operation.
 */
// Codeforces
// 282A - Bit++

#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main() {
  int numStatements = 0;
  // The '\n' in the format string consumes the newline character after the number.
  scanf("%d\n", &numStatements);

  string s = "";
  int total = 0;
  /**
   * @brief Main loop to process each operation.
   * Pre-condition: numStatements holds the number of operations. total is 0.
   * Invariant: After each iteration, 'total' is updated based on the operation.
   */
  while (numStatements--) {
    getline(cin, s);
    // This check works because both "++X" and "X++" have '+' at index 1.
    // Similarly, "--X" and "X--" have '-' at index 1.
    if (s[1] == '+')
      total++;
    else
      total--;
  }

  printf("%d\n", total);

  return 0;
}
