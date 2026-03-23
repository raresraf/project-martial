/**
 * @file 112A.CPP
 * @brief This competitive programming solution performs a case-insensitive
 *        lexicographical comparison between two input strings.
 *        It outputs 1 if the first string is lexicographically greater,
 *        -1 if the second string is greater, and 0 if they are equal.
 * Algorithm: String comparison with case normalization.
 *            1. Read two input strings.
 *            2. Convert both strings to lowercase to ensure case-insensitive comparison.
 *            3. Perform a direct lexicographical comparison on the normalized strings.
 * Time Complexity: O(L), where L is the maximum length of the two input strings.
 *                  This involves iterating through each string once for case conversion
 *                  and then once for the comparison.
 * Space Complexity: O(L), for storing the two input strings. The `transform` function
 *                   modifies the strings in-place, so no additional storage proportional
 *                   to L is allocated.
 */
/*
 * Problem description
 *
 * https://codeforces.com/problemset/problem/112/A
 *
 */

#include <bits/stdc++.h> // Includes common standard library headers, including iostream, string, and algorithm.
using namespace std;

/**
 * @brief Main function of the program. Reads two strings, normalizes their case, compares them, and prints the result.
 * Functional Utility: Serves as the entry point for solving the case-insensitive string comparison problem.
 */
int main() {
  int a;          // Integer to store the comparison result (1, -1, or 0).
  string i1, i2;  // Strings to store the two input words.
  cin >> i1 >> i2; // Reads the two strings from standard input.
  
  // Functional Utility: Converts string 'i1' to all lowercase characters.
  // Invariant: After this operation, 'i1' contains the lowercase version of its original content.
  transform(i1.begin(), i1.end(), i1.begin(), ::tolower);
  // Functional Utility: Converts string 'i2' to all lowercase characters.
  // Invariant: After this operation, 'i2' contains the lowercase version of its original content.
  transform(i2.begin(), i2.end(), i2.begin(), ::tolower);
  
  // Block Logic: Performs a lexicographical comparison of the two lowercase strings.
  // If 'i1' is greater than 'i2', 'a' becomes 1.
  // Else if 'i1' is less than 'i2', 'a' becomes -1.
  // Otherwise (if they are equal), 'a' becomes 0.
  a = i1 > i2 ? 1 : (i1 < i2 ? -1 : 0);
  
  cout << a; // Prints the integer result of the comparison.
  return 0;  // Indicates successful program termination.
}
