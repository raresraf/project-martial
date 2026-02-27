/**
 * @file 112A.CPP
 * @brief Solution for competitive programming problem: "Petya and Strings".
 *
 * @details This program compares two input strings lexicographically, ignoring
 *          case. It first converts both strings to lowercase and then performs
 *          a standard string comparison. The output is -1 if the first string
 *          is lexicographically smaller, 1 if it's larger, and 0 if they are equal.
 *          Algorithm: Convert to lowercase and perform direct string comparison.
 *          Time Complexity: O(L), where L is the length of the shorter string,
 *                           as characters are processed once for conversion and once for comparison.
 *          Domain: Competitive Programming.
 */
// https://codeforces.com/problemset/problem/112/A
// 112 A. Petya and Strings

#include <iostream>
using namespace std;
int main() {
  string a, b; // Declare strings 'a' and 'b' to store the two input strings.
  cin >> a >> b; // Read the two strings from standard input.
  
  // Block Logic: Convert both strings to lowercase for case-insensitive comparison.
  // Invariant: After this loop, both strings 'a' and 'b' will contain only lowercase characters.
  for (int i = 0; i < a.size(); i++) {
    a[i] = tolower(a[i]); // Convert character in 'a' to lowercase.
    b[i] = tolower(b[i]); // Convert character in 'b' to lowercase.
  }
  
  // Block Logic: Perform lexicographical comparison of the lowercase strings.
  if (a > b) // If string 'a' is lexicographically greater than string 'b'.
    cout << 1;
  else if (a == b) // If string 'a' is lexicographically equal to string 'b'.
    cout << 0;
  else // If string 'a' is lexicographically smaller than string 'b'.
    cout << -1;
}
