/**
 * @file 112A.CPP
 * @brief Lexicographical comparison of two strings.
 *
 * This program reads two strings, converts them to lowercase, and then
 * compares them lexicographically. It prints -1 if the first string is
 * smaller, 1 if the second string is smaller, and 0 if they are equal.
 *
 * @b Algorithm: The core logic involves a character-by-character comparison of the two input strings after normalization to lowercase.
 * @b Time_Complexity: O(L) where L is the length of the strings, as it iterates through the strings once.
 * @b Space_Complexity: O(L) for storing the two input strings.
 */
#include <cstdio>
#include <iostream>
#include <string.h>

using namespace std;

/**
 * @brief Converts a string to lowercase.
 * @param ch The input string.
 * @return The lowercase version of the string.
 *
 * This function iterates through each character of the string and converts it to lowercase.
 */
string convert_lower(string ch) {
  // Loop through each character of the string.
  for (int i = 0; i < ch.size(); i++) {
    // Convert the character to lowercase.
    ch[i] = tolower(ch[i]);
  }

  return ch;
}

int main() {
  string first, second;
  // Read the two strings from standard input.
  cin >> first >> second;
  // Convert both strings to lowercase for case-insensitive comparison.
  first = convert_lower(first);
  second = convert_lower(second);
  int res = 0;
  /**
   * @brief Compare the two strings character by character.
   *
   * This loop iterates through the strings and compares them lexicographically.
   * If a difference is found, the result is set and the loop is terminated.
   */
  for (int i = 0; i < first.size(); i++) {
    // If characters are the same, continue to the next character.
    if (first[i] == second[i]) {

      continue;
    } else if (first[i] > second[i]) {
      // If the character in the first string is greater, set result to 1.
      res = 1;
      break;
    } else {
      // If the character in the second string is greater, set result to -1.
      res = -1;
      break;
    }
  }
  // Print the result of the comparison.
  cout << res;
  return 0;
}