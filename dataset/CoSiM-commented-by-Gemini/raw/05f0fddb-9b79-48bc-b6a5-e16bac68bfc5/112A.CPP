/**
 * @file 112A.CPP
 * @brief Solution for competitive programming problem "Petya and Strings".
 * @details This program performs a case-insensitive lexicographical comparison
 * of two input strings. It converts both strings to lowercase and then
 * determines their alphabetical order.
 *
 * Algorithm:
 * 1. Read two strings from standard input.
 * 2. Define a helper function `lowerCase` that converts a given string to all
 *    lowercase characters.
 * 3. Apply the `lowerCase` function to both input strings.
 * 4. Use the standard string comparison operators (`>`, `<`, `==`) to compare
 *    the two lowercase strings lexicographically.
 * 5. Print '1' if the first string is lexicographically greater than the second.
 * 6. Print '-1' if the second string is lexicographically greater than the first.
 * 7. Print '0' if the strings are identical.
 *
 * Time Complexity: O(L), where L is the length of the strings. This is
 * dominated by the string conversion and comparison operations.
 * Space Complexity: O(L) to store the two input strings.
 */
#include <iostream>

using namespace std;
/**
 * @brief Converts a string to its lowercase equivalent.
 * @param str The input string.
 * @return A new string with all characters converted to lowercase.
 */
string lowerCase(string str) {
  for (int i = 0; i < str.length(); i++)
    str[i] = tolower(str[i]);
  return str;
}
int main() {
  string first, second;
  // Read the two strings to be compared.
  cin >> first >> second;

  // Convert both strings to lowercase for case-insensitive comparison.
  first = lowerCase(first);
  second = lowerCase(second);
  // Perform lexicographical comparison on the lowercase strings.
  if (first > second)
    cout << "1" << endl;
  else if (second > first)
    cout << "-1" << endl;
  else
    cout << "0" << endl;

  return 0;
}
