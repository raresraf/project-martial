//
//  339A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

/**
 * @file 339A.cpp
 * @brief Solution to the "Helpful Maths" competitive programming problem.
 * @details The problem requires reordering the terms in a sum of single-digit numbers
 * into non-decreasing order. For example, an input "3+2+1" should result in "1+2+3".
 *
 * @section algorithm Algorithm
 * The approach is to treat the numbers as characters and leverage string manipulation and sorting:
 * 1. Read the entire sum as a single string `s`.
 * 2. Create a new string `out` to hold only the digits.
 * 3. Iterate through the input string `s`. The digits are always at even indices (0, 2, 4, ...),
 *    so append these characters to the `out` string.
 * 4. Sort the `out` string lexicographically. Since the characters are digits '1'-'9', this
 *    effectively sorts the numbers they represent.
 * 5. Iterate through the sorted `out` string and print each digit, followed by a '+' sign,
 *    ensuring that no trailing '+' is printed after the last digit.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L log L), where L is the number of digits in the sum. This is dominated by the sorting step. The length of the input string is at most 2L-1.
 * - Space Complexity: O(L) to store the extracted digits in the `out` string.
 */
#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s, out;

  // Read the input sum as a string (e.g., "3+1+2").
  cin >> s;

  // Extract only the digits from the input string.
  // The summands are at even positions (0, 2, 4, ...).
  for (int i = 0; i < s.length(); ++i)
    if (i % 2 == 0)
      out += s[i];

  // Sort the string containing the digits. Since character codes for '1' through '9'
  // are ordered, this correctly sorts the numbers.
  sort(out.begin(), out.end());

  // Print the sorted numbers, reformatted as a sum.
  for (int i = 0; i < out.length(); ++i) {
    // Use a ternary operator to print '+' between numbers, but not after the last one.
    i != out.length() - 1 ? cout << out[i] << '+' : cout << out[i];
  }
}
