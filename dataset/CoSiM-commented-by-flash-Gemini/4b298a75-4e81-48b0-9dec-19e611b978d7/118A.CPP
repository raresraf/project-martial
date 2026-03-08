/**
 * @4b298a75-4e81-48b0-9dec-19e611b978d7/118A.CPP
 * @brief This program processes an input string, converting it to lowercase
 *        and then printing each consonant prefixed with a ".".
 * Algorithm: Iterative character processing, checking against a set of vowels.
 * Time Complexity: O(L) where L is the length of the input string, due to a single pass.
 * Space Complexity: O(L) for storing the input string.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s; // Declare a string variable 's' to store the input word.
  // Functional Utility: Read the input string.
  cin >> s;
  // Block Logic: Iterate through each character of the input string.
  for (int i = 0; i < s.length(); i++) {
    // Functional Utility: Convert the current character to lowercase for case-insensitive comparison.
    char c = tolower(s[i]);
    // Pre-condition: Check if the character is a vowel (a, o, u, y, e, i).
    if (c == 'a' || c == 'o' || c == 'u' || c == 'y' || c == 'e' || c == 'i')
      continue; // Action: Skip vowels.
    else
      // Action: If it's a consonant, print a "." followed by the lowercase consonant.
      cout << "." << c;
  }
  return 0; // Indicate successful program termination.
}
