/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem using standard library functions.
 *
 * @details
 * The program capitalizes the first letter of a given word. This implementation
 * is clean and portable as it uses the standard C++ library functions `islower`
 * and `toupper`.
 *
 * Algorithm:
 * 1. Read the input word.
 * 2. Use `islower()` to check if the first character is a lowercase letter.
 * 3. If it is, use `toupper()` to convert it to uppercase in-place.
 * 4. Print the modified string.
 *
 * Time Complexity: O(L) where L is the length of the string, due to I/O.
 *                  The capitalization logic itself is O(1).
 * Space Complexity: O(L) to store the string.
 */
#include <bits/stdc++.h>
#include <cctype>
#include <iostream>
using namespace std;

int main() {
  string str;
  cin >> str;
  // Check if the first character is a lowercase letter using the standard library function.
  if (islower(str[0])) {
    // If it is, convert it to uppercase.
    str[0] = toupper(str[0]);
  }
  cout << str;
  return 0;
}
