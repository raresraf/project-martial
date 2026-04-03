/**
 * @file 281A.cpp
 * @brief A concise solution to the "Word Capitalization" problem.
 * This program reads a word and capitalizes its first letter using the `toupper()`
 * function, which is a robust way to handle character case conversion.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Use `toupper()` to convert the first character of the string to uppercase.
 *    If the character is already uppercase or not a letter, it remains unchanged.
 * 3. Print the modified string.
 *
 * Time Complexity: O(L) for reading the string, where L is the length of the string. The capitalization is O(1).
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  // Read the input word.
  cin >> a;
  // Capitalize the first character using the toupper function.
  a[0] = toupper(a[0]);
  // Print the modified word.
  cout << a;
}
