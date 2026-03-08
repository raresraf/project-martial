/**
 * @499c17d5-8f6b-4934-8156-0003abfae069/281A.CPP
 * @brief This program capitalizes the first letter of an input word while
 *        leaving the rest of the characters unchanged.
 * Algorithm: Direct string manipulation, converting the first character to uppercase.
 * Time Complexity: O(L) where L is the length of the input word, as it accesses and potentially modifies a single character.
 * Space Complexity: O(L) for storing the input string.
 */
#include <cctype>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s; // Declare a string variable 's' to store the input word.
  // Functional Utility: Read the input word.
  cin >> s;
  // Functional Utility: Convert the first character of the string to uppercase.
  // 's[0]' accesses the first character, and 'toupper()' converts it to its uppercase equivalent.
  s[0] = toupper(s[0]);
  // Functional Utility: Output the capitalized word.
  cout << s << endl;
  return 0; // Indicate successful program termination.
}