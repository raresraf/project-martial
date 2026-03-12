/**
 * @5cf8d9f0-6446-4950-912c-1fe5f10e7c4e/118A.CPP
 * @brief This program transforms an input string by applying a series of rules:
 * 1. It identifies and replaces all vowels (case-insensitively) with a placeholder character ('0').
 * 2. It then iterates through the modified string, converts any remaining uppercase
 *    consonants to lowercase, and prints each consonant prefixed with a dot ('.').
 *
 * Algorithm: Two-pass string processing. First pass marks vowels. Second pass prints consonants with formatting.
 * Time Complexity: O(L) where L is the length of the input string (due to string operations and loops).
 * Space Complexity: O(L) for storing the modified string.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, string, cctype.
#include <cstring>       // Included but not directly used in this snippet.
using namespace std;     // Uses the standard namespace.

int main() {
  string s, vowels = "aoyeui", rp = "0"; // Declares input string 's', 'vowels' string, and 'rp' for replacement.
  cin >> s;       // Reads the input string from standard input.
  int n = s.length(); // Gets the length of the input string.
  
  /**
   * Block Logic: First pass - iterates through the input string, identifies vowels (case-insensitively),
   * and replaces them with the placeholder character '0'.
   * Invariant: After this loop, 's' contains '0's where vowels used to be, and consonants remain.
   */
  for (int i = 0; i < n; i++) { // Loops through each character of the string.
    // Inline: Converts the character to lowercase and checks if it's found in the 'vowels' string.
    // The condition `vowels.find(...) >= 0 && vowels.find(...) <= n` is a way to check if `string::npos` was NOT returned.
    if (vowels.find(tolower(s[i])) >= 0 && vowels.find(tolower(s[i])) <= n) { 
      s.replace(i, 1, rp); // Inline: Replaces the vowel at current position 'i' with "0".
    }
  }
  
  /**
   * Block Logic: Second pass - iterates through the modified string 's'.
   * For each character that is not a placeholder '0' (i.e., a consonant),
   * it converts it to lowercase if necessary and prints it prefixed with a dot.
   * Invariant: The transformed string (consonants prefixed with dots) is printed to output.
   */
  for (int i = 0; i < n; i++) { // Loops through each character of the modified string.
    if (s[i] != '0') { // Inline: Checks if the character is not the vowel placeholder.
      // Inline: Checks if the character is an uppercase letter (ASCII A-Z).
      if (s[i] > 64 && s[i] < 91) 
        cout << "." << (char)(s[i] + 32); // Prints dot, then converts to lowercase and prints.
      else // Inline: Character is a lowercase consonant.
        cout << "." << s[i]; // Prints dot, then the character.
    }
  }
  return 0; // Indicates successful program termination.
}
