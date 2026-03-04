/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two strings using C-style arrays.
 *
 * This program reads two strings, converts them to lowercase, and then compares
 * them lexicographically. It's a C-style implementation of a common competitive
 * programming problem, using character arrays and functions like `strcmp`.
 *
 * @algorithm Case normalization followed by direct lexicographical comparison.
 * @time_complexity O(L), where L is the length of the strings, due to the
 *                  loop for case conversion and the `strcmp` function.
 * @space_complexity O(L) to store the two input strings in character arrays.
 */
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

/**
 * @brief Converts an uppercase character to lowercase.
 *
 * @param in The input character.
 * @return The lowercase equivalent of the character if it's uppercase,
 *         otherwise the character itself.
 */
char easytolower(char in) {
  if (in <= 'Z' && in >= 'A')
    return in - ('Z' - 'z');
  return in;
}

/**
 * @brief Main function to perform the string comparison.
 */
int main() {
  char a[105], b[105];
  // Read two strings into character arrays.
  cin >> a >> b;
  int len = strlen(a);

  /**
   * @brief Normalizes both strings to lowercase.
   * This loop iterates through each character of the arrays and applies
   * the custom `easytolower` function.
   */
  for (int i = 0; i < len; i++) {
    a[i] = easytolower(a[i]);
    b[i] = easytolower(b[i]);
  }

  /**
   * @brief Compares the normalized strings and outputs the result.
   * `strcmp` returns:
   *  - A negative value if string 'a' is less than string 'b'.
   *  - A positive value if string 'a' is greater than string 'b'.
   *  - Zero if they are equal.
   * The program directly prints this integer result.
   */
  cout << strcmp(a, b);
  return 0;
}