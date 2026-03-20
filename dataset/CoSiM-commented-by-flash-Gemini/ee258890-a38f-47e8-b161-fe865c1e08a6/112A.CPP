/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two input strings.
 *        The result indicates which string comes first in dictionary order, or if they are identical.
 * Algorithm: String transformation followed by built-in string comparison.
 *            Both strings are converted to lowercase, then their lexicographical
 *            order is determined. The raw comparison result is normalized to -1, 0, or 1.
 * Time Complexity: O(L) where L is the length of the longer string, due to linear-time
 *                  character transformations and string comparison.
 * Space Complexity: O(L) for storing the input strings and their transformed versions.
 */
// Problem Link: https://codeforces.com/problemset/problem/112/A

#include <iostream>  // For standard input/output operations (cin, cout).
#include <string>    // For string manipulation (std::string).
#include <algorithm> // Required for std::transform.
#include <cctype>    // Required for ::tolower.
#include <cmath>     // Required for std::abs.
#include <bits/stdc++.h> // A common competitive programming header, includes many standard libraries.

using namespace std;

/**
 * @brief Main function to read strings, compare them case-insensitively, and print the result.
 * Functional Utility: Provides the entry point for comparing two strings after
 *                     normalizing their case, indicating their lexicographical order.
 */
int main() {
  string str1, str2; // Declares two string variables to hold the input.

  // Block Logic: Reads the first string from standard input.
  cin >> str1;
  // cout << str1; // Commented-out debug line to print the first string.

  // Block Logic: Reads the second string from standard input.
  cin >> str2;
  // cout << str2; // Commented-out debug line to print the second string.

  int comp; // Variable to store the raw result of the string comparison.

  // Block Logic: Converts all characters in str1 to lowercase.
  // Functional Utility: Ensures a case-insensitive comparison by normalizing the string.
  // Pre-condition: `str1` is a valid string.
  transform(str1.begin(), str1.end(), str1.begin(), ::tolower);

  // Block Logic: Converts all characters in str2 to lowercase.
  // Functional Utility: Ensures a case-insensitive comparison by normalizing the string.
  // Pre-condition: `str2` is a valid string.
  transform(str2.begin(), str2.end(), str2.begin(), ::tolower);

  // Functional Utility: Performs a lexicographical comparison between the two lowercase strings.
  //                     Returns 0 if equal, a negative value if str1 < str2, and a positive
  //                     value if str1 > str2. The magnitude of non-zero values is unspecified.
  comp = str1.compare(str2);

  // Block Logic: Normalizes the comparison result to -1, 0, or 1.
  // Functional Utility: Converts any non-zero comparison result to either -1 or 1,
  //                     providing a standard output format for lexicographical order.
  if (comp != 0) {
    comp = comp / (abs(comp));
  }

  // Functional Utility: Prints the normalized comparison result.
  // Output: -1 if str1 < str2, 0 if str1 == str2, 1 if str1 > str2.
  cout << comp;

  return 0; // Indicates successful program termination.
}