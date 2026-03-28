/**
 * @file 112A.CPP
 * @brief Performs case-insensitive lexicographical comparison of two strings.
 *
 * This program solves the "Petya and Strings" problem. It compares two strings
 * lexicographically, ignoring differences in case. The output is -1 if the first
 * string is smaller, 1 if the second string is smaller, and 0 if they are equal.
 *
 * @b Algorithm: The program first converts both input strings to lowercase. It then
 *    uses the standard string comparison function. The result of the comparison
 *    is then normalized to -1, 0, or 1.
 * @b Time_Complexity: O(L), where L is the length of the strings, as both the
 *    case conversion and the comparison take linear time.
 * @b Space_Complexity: O(1) of extra space, as the strings are modified in-place.
 */
#include <iostream>
#include <string>
#include <bits/stdc++.h> // A common header in competitive programming for including all standard libraries.

using namespace std;

int main()
{
  string str1, str2;
  cin >> str1;
  cin >> str2;

  int comp;

  // Block Logic: Convert both strings to lowercase for case-insensitive comparison.
  // `transform` applies the `tolower` function to every character in the string.
  transform(str1.begin(), str1.end(), str1.begin(), ::tolower);
  transform(str2.begin(), str2.end(), str2.begin(), ::tolower);

  // Perform the lexicographical comparison on the lowercase strings.
  comp = str1.compare(str2);

  // Pre-condition: Normalize the result to be exactly -1, 0, or 1.
  if (comp != 0) {
    // This is a mathematical trick to get the sign of a number.
    // If comp is negative, comp/abs(comp) is -1.
    // If comp is positive, comp/abs(comp) is 1.
    comp = comp / (abs(comp));
  }
  
  cout << comp;
}
