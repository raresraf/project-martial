/**
 * @file 282A.CPP
 * @brief Solution to Codeforces problem 282A - Bit++.
 * @description This program simulates a simple programming language with one variable, `X`,
 *              and two operations: increment (++) and decrement (--). Given a sequence of
 *              statements like "++X", "X--", etc., it calculates the final value of `X`
 *              starting from an initial value of 0.
 *
 * Algorithm: The program initializes a counter variable to 0. It then reads the number of
 *            statements to execute. In a loop, it reads each statement as a string and
 *            inspects its characters to determine if it's an increment or decrement
 *            operation. A simpler approach is to check for the '+' character anywhere
 *            in the string, as its presence always signifies an increment, and its
 *            absence (implying the presence of '-') signifies a decrement. This code
 *            explicitly checks the first two characters to identify the operation.
 *
 * Time Complexity: O(N), where N is the number of statements, as it processes each
 *                  statement once.
 * Space Complexity: O(1), as the space used does not scale with the number of statements
 *                   (the string for each statement is reused).
 */
#include <iostream>
using namespace std;

string b; // Stores the current statement string.
// 'a' is the number of statements, 'c' is the variable 'X', initialized to 0 globally.
int a, c; 

int main() {
  // Read the total number of statements.
  cin >> a;
  /**
   * Block Logic: Loop through each statement and update the counter 'c'.
   */
  for (int i = 0; i < a; i++) {
    cin >> b;
    // The if-else block checks the format of the statement to determine
    // whether it's an increment or a decrement.
    if (b[0] == '+') { // Covers "++X"
      c++;
    } else if (b[0] == '-') { // Covers "--X"
      c--;
    } else if (b[0] == 'X' && b[1] == '-') { // Covers "X--"
      c--;
    } else if (b[0] == 'X' && b[1] == '+') { // Covers "X++"
      c++;
    }
  }
  // Print the final value of the variable.
  cout << c;
}
