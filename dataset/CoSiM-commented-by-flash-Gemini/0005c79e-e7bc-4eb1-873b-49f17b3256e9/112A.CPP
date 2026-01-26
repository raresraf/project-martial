/**
 * @file 112A.CPP
 * @brief Implements a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings, converts them to lowercase, and then
 * compares them character by character to determine their lexicographical order.
 *
 * Algorithm: Iterative character-by-character comparison after converting to lowercase.
 * Time Complexity: O(N) where N is the length of the shorter string.
 * Space Complexity: O(N) for storing the strings.
 */

#include <cstdio>
#include <iostream>
#include <string.h>

using namespace std;

/**
 * @brief Converts a given string to its lowercase equivalent.
 * @param ch The input string to convert.
 * @return The lowercase version of the input string.
 *
 * Functional Utility: Ensures case-insensitive comparison by normalizing string case.
 * Invariant: The returned string will contain only lowercase characters.
 */
string convert_lower(string ch) {
  // Iterates through each character of the string.
  for (int i = 0; i < ch.size(); i++) {
    // Converts the current character to lowercase.
    ch[i] = tolower(ch[i]);
  }

  return ch;
}

int main() {
  string first, second;
  // Reads the first and second strings from standard input.
  cin >> first >> second;
  
  // Converts both strings to lowercase for case-insensitive comparison.
  first = convert_lower(first);
  second = convert_lower(second);
  
  // 'res' stores the comparison result: 0 for equal, 1 if first > second, -1 if first < second.
  int res = 0;
  
  /**
   * Block Logic: Performs a lexicographical comparison of the two strings.
   * Invariant: 'res' accurately reflects the lexicographical order of the strings
   *            up to the current character being compared.
   */
  for (int i = 0; i < first.size(); i++) {
    // If characters are equal, continue to the next character.
    if (first[i] == second[i]) {
      continue;
    } 
    // If the character in 'first' is greater, 'first' is lexicographically larger.
    else if (first[i] > second[i]) {
      res = 1;
      break; // No need to compare further, order is determined.
    } 
    // If the character in 'first' is smaller, 'first' is lexicographically smaller.
    else {
      res = -1;
      break; // No need to compare further, order is determined.
    }
  }
  
  // Prints the final comparison result.
  cout << res;
  return 0;
}
