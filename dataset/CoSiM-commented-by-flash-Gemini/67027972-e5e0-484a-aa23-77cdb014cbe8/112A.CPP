/**
 * @67027972-e5e0-484a-aa23-77cdb014cbe8/112A.CPP
 * @brief This program solves Codeforces problem 112A, "Petya and Strings".
 * It reads two strings, converts them both to lowercase, and then performs a
 * lexicographical comparison between them. It outputs -1 if the first string
 * is lexicographically smaller, 1 if it is larger, and 0 if they are equal.
 *
 * Algorithm: String conversion and lexicographical comparison.
 * Time Complexity: O(L) where L is the maximum length of the two input strings,
 * due to string traversal for conversion and comparison.
 * Space Complexity: O(L) for storing the two input strings.
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers, including <iostream>, <string>, and <algorithm>.
using namespace std;     // Uses the standard namespace.

/**
 * @brief Main function to perform case-insensitive lexicographical comparison of two strings.
 * Reads two strings, converts them to lowercase, compares them, and prints the result (-1, 0, or 1).
 * @return 0 on successful execution.
 */
int main() {
  string s1, s2; // Declares two string variables to store the input strings.
  cin >> s1 >> s2; // Reads the two strings from standard input.
  
  // Inline: Converts all characters in `s1` to lowercase.
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower); 
  // Inline: Converts all characters in `s2` to lowercase.
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower); 
  
  /**
   * Block Logic: Compares the two lowercase strings lexicographically and prints the result.
   * C++ `std::string` comparison operators (`<`, `==`, `>`) perform lexicographical comparison.
   * Pre-condition: `s1` and `s2` are converted to lowercase.
   * Invariant: Prints -1, 0, or 1 based on the comparison result.
   */
  if (s1 < s2) // If `s1` is lexicographically smaller than `s2`.
    cout << "-1\n"; // Prints -1.
  else if (s1 == s2) // If `s1` is lexicographically equal to `s2`.
    cout << "0\n"; // Prints 0.
  else // If `s1` is lexicographically greater than `s2`.
    cout << "1\n"; // Prints 1.
  
  return 0; // Indicates successful program termination.
}