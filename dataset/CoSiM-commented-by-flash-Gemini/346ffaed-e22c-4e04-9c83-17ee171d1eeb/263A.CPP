/**
 * @file 263A.CPP
 * @brief Solution for a competitive programming problem to convert decimal numbers to their binary representation.
 *
 * Algorithm: Iterative division by 2 to extract binary digits, then reversing the order for correct representation.
 * Time Complexity: O(N * log_2(X_max)), where N is the number of test cases and X_max is the maximum input value for X.
 *                  Each conversion takes O(log_2(X)) time.
 * Space Complexity: O(log_2(X_max)) for storing the binary string for the largest number.
 */

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/

/**
 * @brief Main entry point of the program.
 *
 * Functional Utility: Reads an integer `N` representing the number of test cases.
 *                     For each test case, it reads an integer `X` and converts it
 *                     to its binary representation, printing the result to standard output.
 *                     Handles the special case where `X` is 0.
 *
 * @return 0 on successful execution.
 */
int main() {
  int N; // @brief Number of test cases to process.
  cin >> N;
  cin.ignore();
  // Block Logic: Iterate through each of the provided test cases.
  // Invariant: The loop continues `N` times, processing one decimal-to-binary conversion per iteration.
  for (int i = 0; i < N; i++) {
    int X; // @brief The decimal number to be converted to binary.
    cin >> X;
    cin.ignore();
    string s = ""; // @brief String to store the binary representation (in reverse order).
    // Block Logic: Handle the special case where the input number `X` is 0.
    // Invariant: If `X` is 0, print 0 directly, as its binary representation is simply "0".
    if (X == 0) {
      cout << X;
    } else {
      // Block Logic: Convert the decimal number `X` to binary using iterative division by 2.
      // Invariant: `s` accumulates the binary digits in reverse order until `X` becomes 0.
      while (X) {

        s += X % 2; // Inline: Append the remainder (0 or 1) of X divided by 2 to the string.
        X = X / 2; // Inline: Update X to the quotient of X divided by 2.
      }

      // Block Logic: Print the binary string in correct (forward) order.
      // Invariant: Iterates from the last character to the first, printing each binary digit.
      for (int i = s.length() - 1; i >= 0; i--) {
        cout << s[i];
      }
    }
    cout << endl; // Functional Utility: Print a newline after each binary output for formatting.
  }

  // Functional Utility: Placeholder comments from auto-generated code, no functional impact on the solution logic.
  // Write an answer using cout. DON'T FORGET THE "<< endl"
  // To debug: cerr << "Debug messages..." << endl;

  // cout << "ones and zeros" << endl;
}