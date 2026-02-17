/**
 * @file 112A.CPP
 * @brief Solution for the "Petya and Strings" competitive programming problem.
 * @author Tico
 * @date 3/17/17
 *
 * @details This program compares two input strings lexicographically, ignoring
 * case differences. It outputs -1 if the first string is smaller, 1 if the
 * second string is smaller, and 0 if they are equal.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read two strings from standard input.
 * 2. Convert both strings entirely to a consistent case (lowercase).
 * 3. Use the standard string comparison operator to determine their
 *    lexicographical order.
 * 4. Print -1, 1, or 0 based on the result.
 *
 * @note The code contains a flawed and irrelevant summation logic. It calculates
 * the sum of character values for each string, but these sums are never used to
 * determine the final output. The program works correctly only because the
 * final comparison is performed on the properly lowercased strings.
 *
 * @section complexity
 * Time Complexity: O(L), where L is the length of the strings. This is
 * dominated by the loops for case conversion and the final string comparison.
 * Space Complexity: O(L), to store the two input strings.
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  string s1, s2;
  // These summation variables are calculated but are irrelevant to the final output.
  int sum_s1 = 0, sum_s2 = 0;

  cin >> s1;

  // Block Logic: Convert the first string to lowercase.
  for (int i = 0; i < s1.length(); ++i) {
    s1[i] = tolower(s1[i]);
    // This summation is incorrect for lexicographical comparison and is unused.
    sum_s1 += (s1[i] - '0');
  }

  cin >> s2;

  // Block Logic: Convert the second string to lowercase.
  for (int i = 0; i < s2.length(); ++i) {
    s2[i] = tolower(s2[i]);
    // This summation is also incorrect and unused.
    sum_s2 += (s2[i] - '0');
  }

  // Block Logic: Perform case-insensitive lexicographical comparison.
  // This comparison works correctly because both strings have been converted
  // to the same case. The std::string operator< handles the lexicographical check.
  if (s1 < s2)
    cout << "-1";
  else if (s2 < s1)
    cout << "1";
  else
    cout << "0";
}
