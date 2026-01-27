/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces.
 * @details This program reads a single word and capitalizes its first letter.
 * All other letters remain unchanged.
 *
 * @see https://codeforces.com/problemset/problem/281/A
 *
 * Algorithm:
 * 1. Read the input word into a string.
 * 2. Access the first character of the string (at index 0).
 * 3. Convert this character to its uppercase equivalent using `toupper`.
 * 4. Print the modified string.
 *
 * Time Complexity: O(1), as the operation is independent of the string length (it only modifies the first character).
 * Space Complexity: O(L), where L is the length of the input string to store it.
 */
#include <iostream>
#include <locale> // Required for toupper
using namespace std;

int main() {

  string str;
  // Read the word from standard input.
  cin >> str;

  // Access the first character (at index 0) and convert it to uppercase.
  // The toupper function is locale-sensitive.
  str[0] = toupper(str[0]);

  // Print the resulting string.
  cout << str << endl;

  return 0;
}