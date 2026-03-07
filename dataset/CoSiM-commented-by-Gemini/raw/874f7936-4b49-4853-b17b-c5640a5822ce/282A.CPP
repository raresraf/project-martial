/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces (282A).
 *
 * @details
 * The program calculates the final value of a variable 'X' which starts at 0.
 * It is given a sequence of 'n' operations, which can be any of:
 * "++X", "X++", "--X", or "X--".
 *
 * Algorithm:
 * 1. Initialize a counter to 0.
 * 2. Read the number of operations, n.
 * 3. Loop n times:
 *    a. Read the operation as a string.
 *    b. Use a series of `if-else if` statements to identify the operation type.
 *    c. The logic checks specific characters in the string to determine whether
 *       to increment or decrement the counter.
 * 4. Print the final value of the counter.
 *
 * The implementation's conditional logic is convoluted. A simpler approach would be
 * to just check the middle character (at index 1), as it is '+' for all increments
 * and '-' for all decrements.
 *
 * Time Complexity: O(N), where N is the number of operations.
 * Space Complexity: O(1), as each string is processed on the fly.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, count = 0; // count represents the value of variable X, starting at 0.
  cin >> n;
  for (int i = 0; i < n; i++) {
    string x;
    cin >> x;
    // The series of checks below determines if the operation is an increment or decrement.
    if (x[0] == '+' && x[1] == '+') // Handles "++X"
      count++;
    else if (x[0] == '-' || x[1] == '-') // Handles both "--X" and "X--"
      count--;
    else if (x[1] == '+' && x[2] == '+') // Handles "X++"
      count++;
    else if (x[1] == '-' && x[2] == '-') // This case is unreachable due to the second else-if.
      count--;
  }
  cout << count << endl;
  return 0;
}