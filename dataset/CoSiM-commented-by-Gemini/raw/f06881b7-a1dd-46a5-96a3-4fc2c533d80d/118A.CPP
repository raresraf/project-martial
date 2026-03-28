/**
 * @file 118A.CPP
 * @brief Processes a string by removing vowels, adding a dot before
 *        consonants, and converting them to lowercase, using regex.
 *
 * This program takes a string, converts it to lowercase, removes all vowels
 * (a, e, i, o, u, y) using a regular expression, and then inserts a '.'
 * before each remaining character (consonant).
 *
 * @b Algorithm:
 * 1. Convert the entire input string to lowercase.
 * 2. Use `std::regex_replace` to remove all characters that match the vowel regex.
 * 3. Iterate through the resulting string of consonants and build a new string
 *    by prepending each consonant with a '.'.
 *
 * @b Time_Complexity: O(L), where L is the length of the string. The regex
 *    operations and loops are proportional to the string length.
 * @b Space_Complexity: O(L) to store the input string, the string after vowel
 *    removal, and the final result string.
 */
#include <iostream>
#include <regex>

using namespace std;

int main() {
  string s;
  cin >> s;

  // Convert the entire string to lowercase.
  for (int i = 0; i < s.size(); i++) {
    s[i] = tolower(s[i]);
  }

  // Use a regular expression to remove all vowels.
  s = regex_replace(s, regex("[aeiouy]"), "");
  
  string res = "";
  /**
   * @brief This loop constructs the final result string.
   * It iterates through the consonants and prepends each with a '.'.
   */
  for (char c : s) {
    res += ".";
    res += c;
  }
  cout << res;
  return 0;
}