/*
        SubmissionId	:	15918774
        ContestId	:	282
        Index	:	A
        ProblemName	:	Bit++
        ProblemTags	:	['implementation']
        ProgrammingLanguage	:	GNU C++
        Verdict	:	OK
*/

/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem using character array inspection.
 *
 * This program simulates a series of increment and decrement operations on a single variable 'x'.
 * It reads an integer N, followed by N operation strings. Instead of string comparison,
 * it inspects characters at specific indices of the input C-style string to determine the operation.
 *
 * @b Algorithm: Direct Simulation
 * @b Time_Complexity: O(N), where N is the number of operations.
 * @b Space_Complexity: O(1), as the character array for the operation is fixed-size.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: the number of operations to perform.
  // x: the variable to be manipulated, initialized to 0.
  long long int n, i, j, x = 0;
  cin >> n;

  // A character array to store the operation string (e.g., "++X", "X--").
  char a[1000];

  /**
   * @brief Main simulation loop, processing one operation at a time.
   * Pre-condition: 'n' contains the total number of operations.
   * Invariant: 'x' holds the net result of the first 'i' operations.
   */
  for (i = 0; i < n; i++) {
    cin >> a;
    /**
     * @brief Determines the operation type by checking for a '+' character.
     * Logic: An operation is an increment if a '+' exists at the first or third position
     * (e.g., "++X" or "X++"). Otherwise, it's a decrement. This avoids full string comparison.
     */
    if (a[0] == '+' || a[2] == '+')
      x += 1;
    else
      x -= 1;
  }

  // Output the final value of x.
  cout << x;
  return 0;
}