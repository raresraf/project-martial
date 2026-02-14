/**
 * @file 339A.cpp
 * @brief Solution for a problem that involves rearranging a sum of numbers.
 * @details The program takes a string representing a sum of single-digit numbers (1, 2, or 3),
 * such as "3+1+2", and outputs the sum with the numbers rearranged in non-decreasing order, e.g., "1+2+3".
 *
 * @section algorithm Algorithm
 * 1. Read the input string, which contains digits separated by '+' signs.
 * 2. Extract only the digits from the string. Since the digits are at even indices (0, 2, 4, ...),
 *    iterate through the input string and append the characters at these indices to a new string.
 * 3. Sort the new string containing just the digits. Since characters are sorted by their ASCII values,
 *    this correctly sorts the digits '1', '2', '3'.
 * 4. Construct the final output string by iterating through the sorted digits and inserting a '+'
 *    sign between each adjacent pair of digits.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L log L), where L is the number of digits in the sum. This is dominated by the sorting step.
 * - Space Complexity: O(L), to store the string of extracted digits.
 */
#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function to execute the sum rearrangement logic.
 * @return 0 on successful execution.
 */
int main() {
  string s, out;

  // Read the string representing the sum.
  cin >> s;

  // Extract the digits (which are at even indices) into the 'out' string.
  for (int i = 0; i < s.length(); ++i) {
    if (i % 2 == 0) {
      out += s[i];
    }
  }

  // Sort the string of digits.
  sort(out.begin(), out.end());

  // Print the sorted digits, separated by '+'.
  for (int i = 0; i < out.length(); ++i) {
    // Print a '+' after each digit except the last one.
    if (i != out.length() - 1) {
      cout << out[i] << '+';
    } else {
      cout << out[i];
    }
  }
  cout << endl;

  return 0;
}
