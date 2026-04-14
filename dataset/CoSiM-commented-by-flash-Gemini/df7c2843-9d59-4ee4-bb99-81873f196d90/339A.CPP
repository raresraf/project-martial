/**
 * @file 339A.CPP
 * @brief This program solves the "Helpful Maths" problem. It takes an input string
 * representing a sum of '1', '2', and '3' digits separated by '+' signs (e.g., "3+2+1")
 * and outputs them in non-decreasing order (e.g., "1+2+3").
 * Algorithm: Extracts all digits, then reconstructs the string by concatenating all '1's,
 * then all '2's, then all '3's, each followed by a '+', and finally removes the trailing '+'.
 * Time Complexity: O(L) where L is the length of the input string, due to multiple passes.
 * Space Complexity: O(L) for storing intermediate strings.
 */
#include <algorithm> // Included in original, though not explicitly used for sorting here.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <string>    // Required for std::string operations.

using namespace std; // Brings all names from the std namespace into the current scope.

int main() {
  string input = ""; // Stores the raw input sum string.
  string s = "";     // Stores only the digits extracted from the input string.
  string x = "";     // Intermediate string to build the sorted sum with trailing '+'.
  string result = ""; // Stores the final sorted sum string.

  // Reads the input sum string (e.g., "1+3+2").
  cin >> input;

  /**
   * Block Logic: Iterates through the raw input string and extracts only the digit characters.
   * Characters that are not '+' are appended to string 's'.
   * Invariant: String 's' contains all digits from 'input' in their original order.
   */
  for (int i = 0; i < input.length(); i++) {
    if (input[i] != '+') {
      s += input[i]; // Appends the digit to 's'.
    }
  }

  /**
   * Block Logic: Constructs the sorted sum string by appending digits in non-decreasing order.
   * It iterates through the extracted digits in 's' multiple times:
   * First, it appends all '1's found in 's' to 'x', each followed by a '+'.
   * Then, it appends all '2's found in 's' to 'x', each followed by a '+'.
   * Finally, it appends all '3's found in 's' to 'x', each followed by a '+'.
   * Invariant: String 'x' contains all digits from 's' sorted, with a trailing '+'.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      x += s[i];   // Appends the digit '1'.
      x += '+';    // Appends a '+' separator.
    }
  }
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '2') {
      x += s[i];   // Appends the digit '2'.
      x += '+';    // Appends a '+' separator.
    }
  }
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '3') {
      x += s[i];   // Appends the digit '3'.
      x += '+';    // Appends a '+' separator.
    }
  }

  /**
   * Block Logic: Removes the trailing '+' from the constructed string 'x'.
   * The loop iterates through the string 'x', but the substring operation and
   * assignment to 'result' happen in every iteration with the same parameters.
   * This makes the loop redundant, as `result` will only store the value from
   * the final iteration. The intent is to take a substring of 'x' that excludes
   * its last character (the extra '+').
   */
  for (int i = 0; i < x.length(); i++) {
    // This line assigns a substring of 'x' (from beginning up to, but not including, the last character) to 'result'.
    // Due to the loop, this assignment is repeated, and only the final one is effective.
    result = x.substr(0, x.length() - 1);
  }

  // Prints the final sorted sum string to standard output.
  cout << result;

  // Standard program exit.
  return 0;
}
