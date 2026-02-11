/*
Bit++

The classic programming language of Bitland is Bit++. This language is so
peculiar and complicated.

The language is that peculiar as it has exactly one variable, called x. Also,
there are two operations:

Operation ++ increases the value of variable x by 1.
Operation -- decreases the value of variable x by 1.
A statement in language Bit++ is a sequence, consisting of exactly one operation
and one variable x. The statement is written without spaces, that is, it can
only contain characters "+", "-", "X". Executing a statement means applying the
operation it contains.

A programme in Bit++ is a sequence of statements, each of them needs to be
executed. Executing a programme means executing all the statements it contains.

You're given a programme in language Bit++. The initial value of x is 0. Execute
the programme and find its final value (the value of the variable when this
programme is executed).
*/

/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem.
 * @author Your Name
 *
 * @details
 * This program simulates the execution of a simple programming language called Bit++,
 * which has one variable `x` and two operations: `++` (increment) and `--` (decrement).
 * The program reads a series of statements and calculates the final value of `x`,
 * which starts at 0.
 *
 * @section algorithm
 * The algorithm iterates through each statement, identifies the operation (`++` or `--`),
 * and updates a counter variable accordingly. The check for the operation is done
 * by looking for the presence of "++" or "--" as substrings within the statement string.
 *
 * @subsection complexity
 * - Time Complexity: O(N), where N is the number of statements, because each statement is processed once.
 * - Space Complexity: O(1), as the space used is constant regardless of the number of statements (if we don't consider the input string).
 */

#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  // Read the number of statements.
  cin >> n;
  int ans = 0; // This variable represents the value of 'x', initialized to 0.
  
  // Loop through each statement.
  while (n--) {
    string s;
    cin >> s;
    // Check for the increment operation. It can be "++X" or "X++".
    // We check for "++" at the beginning or at the second position.
    if (s.substr(0, 2) == "++" || s.substr(1, 2) == "++")
      ans++;
    // Check for the decrement operation. It can be "--X" or "X--".
    // We check for "--" at the beginning or at the second position.
    else if (s.substr(0, 2) == "--" || s.substr(1, 2) == "--")
      ans--;
  }
  // Print the final value of 'x'.
  cout << ans;
}