#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

/**
 * @file 339A.CPP
 * @brief Solution for a string sorting problem, "Helpful Maths".
 *
 * Problem Statement:
 * Given a string representing a sum of numbers (only 1, 2, and 3), rearrange
 * the numbers so that they are in non-decreasing order. For example, the
 * input "3+2+1" should result in "1+2+3".
 *
 * Algorithm:
 * The code implements a multi-pass sorting approach.
 * 1.  **Digit Extraction**: It first iterates through the input string and
 *     extracts all the digits (operands) into a new string `s`, discarding
 *     the '+' operators.
 * 2.  **Sorting**: It then performs a manual, bucket-like sort. It iterates
 *     through the digit string `s` three times:
 *     - First pass: finds all '1's and appends them to a result string `x`,
 *       each followed by a '+'.
 *     - Second pass: does the same for all '2's.
 *     - Third pass: does the same for all '3's.
 *     After these passes, `x` contains the sorted numbers separated by '+', but
 *     with a trailing '+'.
 * 3.  **Formatting**: Finally, it removes the trailing '+' from string `x` to
 *     produce the final `result`.
 *
 * @note This implementation is correct but contains inefficiencies. A more
 * standard approach would be to extract the digits into a `std::vector`,
 * sort the vector using `std::sort`, and then build the output string. The
 * final loop to remove the trailing '+' is also highly redundant.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) to store the intermediate and final strings.
 */

using namespace std;

int main() {
  string input = ""; // The original input string, e.g., "3+1+2".
  string s = "";     // String to hold only the digits, e.g., "312".
  string x = "";     // String to build the sorted output with trailing '+'.
  string result = "";// The final formatted output string.
  cin >> input;

  // Block 1: Extract digits from the input string.
  for (int i = 0; i < input.length(); i++) {
    if (input[i] != '+') {
      s += input[i];
    }
  }

  // Block 2: Sort the digits using a three-pass approach.
  // This is effectively a manual counting/bucket sort for '1', '2', '3'.
  // Pass 1: Collect all '1's.
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      x += s[i];
      x += '+';
    }
  }
  // Pass 2: Collect all '2's.
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '2') {
      x += s[i];
      x += '+';
    }
  }
  // Pass 3: Collect all '3's.
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '3') {
      x += s[i];
      x += '+';
    }
  }

  // Block 3: Remove the trailing '+'.
  // This loop is redundant; it performs the same substring operation
  // multiple times. A single call `x.substr(0, x.length() - 1)` would suffice.
  if (!x.empty()) {
    result = x.substr(0, x.length() - 1);
  }

  cout << result;

  return 0;
}