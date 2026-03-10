/**
 * @file 339A.CPP
 * @brief This program solves the "Xenia and Bushes" problem (Codeforces 339A).
 *   It rearranges a given sum of numbers (1, 2, or 3) into non-decreasing order.
 *   The input is a string representing the sum (e.g., "1+3+2+1"), and the output should be the rearranged sum (e.g., "1+1+2+3").
 *   Problem description: https://codeforces.com/problemset/problem/339/A
 * * Algorithm: Parsing, Sorting, and Formatting.
 *   1. Digits are extracted from the input string by ignoring the '+' characters and stored in a new string.
 *   2. This new string of digits is then sorted in non-decreasing order.
 *   3. The sorted digits are printed, re-inserting '+' signs between them.
 * Time Complexity: O(L + N log N), where L is the length of the input string and N is the number of summands (length of `str2`).
 *   Effectively O(L log L) or O(N log N) since L is proportional to N. For L <= 100, this is very efficient.
 * Space Complexity: O(N) to store the extracted digits in `str2`.
 */
#include <algorithm> // Required for std::sort.
#include <bits/stdc++.h> // Includes most standard C++ libraries.
#include <iostream>  // Required for std::cin, std::cout, std::endl.
#include <string>    // Required for std::string.
using namespace std;

/**
 * @brief Main function to rearrange and print the sum in non-decreasing order.
 * Functional Utility: Reads the input sum string, extracts the numbers, sorts them,
 *                     and then prints the numbers with '+' signs in between.
 */
int main() {
  string str, str2; // `str` for original input, `str2` for extracted digits.
  // Block Logic: Reads the input sum string.
  cin >> str;
  /**
   * Block Logic: Parses the input string `str` to extract only the digit characters.
   * Pre-condition: `str` contains digits and '+' signs.
   * Invariant: `str2` accumulates only the digit characters from `str`.
   */
  for (int i = 0; i < str.size(); i++) {
    if (str[i] != '+') {
      str2.push_back(str[i]); // Appends the digit character to `str2`.
    }
  }
  // Block Logic: Sorts the string `str2` which now contains only the digits.
  // The characters are sorted based on their ASCII values, which correctly sorts '1', '2', '3'.
  sort(str2.begin(), str2.end());
  // Block Logic: Prints the first digit of the sorted sequence without a preceding '+'.
  cout << str2[0];
  /**
   * Block Logic: Iterates through the rest of the sorted digits, printing each with a preceding '+'.
   * Pre-condition: `str2` contains sorted digits, and the first digit has already been printed.
   * Invariant: `cout` outputs the sorted digits separated by '+'.
   */
  for (int i = 1; i < str2.size(); i++) {
    cout << '+' << str2[i];
  }
  // Block Logic: Prints two newline characters. Note that `endl` also flushes the output buffer,
  // potentially being slower than `'\n'` in competitive programming if many outputs are buffered.
  cout << '\n' << endl;

  // cout<<str<<endl; // This line is commented out, likely for debugging or testing.
}
