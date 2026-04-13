/* Problem Statements
*  A. Bit++
time limit per test2 seconds
memory limit per test256 megabytes
inputstandard input
outputstandard output

The classic programming language of Bitland is Bit++.
This language is so peculiar and complicated.

The language is that peculiar as it has exactly one variable, called x.
Also, there are two operations:

Operation ++ increases the value of variable x by 1.
Operation -- decreases the value of variable x by 1.

A statement in language Bit++ is a sequence, consisting of exactly one operation
and one variable x. The statement is written without spaces, that is, it can
only contain characters "+", "-", "X". Executing a statement means applying the
operation it contains.

A programme in Bit++ is a sequence of statements, each of them needs to be
executed. Executing a programme means executing all the statements it contains.

You're given a programme in language Bit++.
The initial value of x is 0. Execute the programme and find its final value (the
value of the variable when this programme is executed).

Input
The first line contains a single integer n (1??n??150) ? the number of
statements in the programme.

Next n lines contain a statement each. Each statement contains exactly one
operation (++ or --) and exactly one variable x (denoted as letter X). Thus,
there are no empty statements. The operation and the variable can be written in
any order.

Output
Print a single integer ? the final value of x.

Examples
input
1
++X
output
1
input
2
X++
--X
output
0
*/

/*
 *  Author   : Byungwook Lee
 *  Date     : 2016/10/24
 *  E-mail   : quddnr153@gmail.com
 *  Github   : github.com/quddnr153
 *  Filename : 282A_bit++.cpp
 */

/**
 * @file 282A.CPP
 * @brief Solution to a competitive programming problem: "A. Bit++".
 *
 * This program simulates the execution of a simple programming language called Bit++.
 * The language operates on a single variable 'x', initially 0, and supports two operations:
 * '++' (increment 'x' by 1) and '--' (decrement 'x' by 1). Statements are
 * 3-character strings like "++X", "X++", "--X", or "X--". The program reads
 * a sequence of these statements and outputs the final value of 'x' after
 * all statements are executed.
 *
 * Algorithm: Iterative statement processing with conditional value update.
 * Time Complexity: O(N), where N is the number of statements. Each statement
 *                  is processed in constant time.
 * Space Complexity: O(1) for storing variables, plus O(S_max) for the maximum
 *                   length of a statement string (which is constant, 3).
 */
#include <iostream> // Required for input/output operations (cin, cout).
#include <string.h> // Includes C-style string functions (though std::string is used).
#include <string>   // Required for std::string and its functionalities.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

/**
 * @brief Main function: entry point of the program.
 * Reads the number of statements, then iteratively processes each statement
 * to update the value of 'x', and finally prints the accumulated value of 'x'.
 */
int main() {
  int N;       // Declares 'N' to store the number of statements in the program.
  int cnt = 0; // Declares 'cnt' to store the current value of variable 'x', initialized to 0.
  cin >> N;    // Reads the total number of statements.

  /**
   * @brief Processes each statement in the Bit++ program.
   * Block Logic: Reads each statement as a string. Based on the presence of
   * a '+' character, it increments or decrements the 'cnt' variable.
   * Pre-condition: 'N' is the total number of statements.
   * Invariant: 'cnt' reflects the value of 'x' after processing all statements up to the current iteration.
   */
  for (int i = 0; i < N; i++) {
    string tmp;   // Declares a temporary string to store the current statement.
    cin >> tmp;   // Reads the current statement.
    // Checks if the statement contains an increment operation.
    // A statement is always 3 characters long (e.g., "++X" or "X++").
    // Checking `tmp[0]` or `tmp[2]` for '+' is sufficient to determine the operation.
    if (tmp[0] == '+' || tmp[2] == '+')
      cnt++; // If it's an increment operation, increment 'cnt'.
    else
      cnt--; // Otherwise (if it contains '--'), it's a decrement operation, decrement 'cnt'.
  }
  cout << cnt << endl; // Prints the final value of 'x' after executing all statements.
  return 0;            // Indicates successful program execution.
}