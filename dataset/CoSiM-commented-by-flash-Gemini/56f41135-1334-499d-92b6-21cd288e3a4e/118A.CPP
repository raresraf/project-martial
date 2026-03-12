/**
 * @56f41135-1334-499d-92b6-21cd288e3a4e/118A.CPP
 * @brief This program implements a text transformation algorithm that modifies an input string.
 * The transformation involves converting all uppercase letters to lowercase, removing specific vowels,
 * and prefixing each remaining consonant with a dot ('.').
 *
 * Algorithm: Single pass character processing with conditional transformation and filtering.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(1) (excluding input string storage, as output is streamed).
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes many standard libraries.
using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {
  int n; // Declared but not used in the provided snippet.
  string s; // Declares a string variable to store the input word.
  cin >> s; // Reads the input string from standard input.
  
  /**
   * Block Logic: Iterates through each character of the input string to apply transformations.
   * Invariant: Characters are processed from left to right, and modifications are applied in-place or streamed.
   */
  for (int i = 0; i < s.length(); i++) {
    /**
     * Block Logic: Converts uppercase letters to lowercase.
     * Pre-condition: 's[i]' is a character from the input string.
     * Invariant: 's[i]' becomes its lowercase equivalent if it was uppercase.
     */
    if (s[i] <= 90) // Inline: Checks if the ASCII value of s[i] corresponds to an uppercase letter (A-Z).
      s[i] += 32; // Inline: Adds 32 to the ASCII value to convert an uppercase letter to its lowercase equivalent.
    
    /**
     * Block Logic: Checks if the character is one of the specified vowels and either skips it or prints it with a dot.
     * Pre-condition: 's[i]' is a lowercase character.
     * Invariant: Only consonants are printed, each preceded by a dot.
     */
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
        s[i] == 'u' || s[i] == 'i') {
      continue; // Inline: If the character is a vowel, it is skipped (not printed), and the loop continues to the next character.
    } else {
      cout << "." << s[i]; // Inline: If the character is a consonant, a dot is printed, followed by the character itself.
    }
  }
  // The program implicitly returns 0 on successful completion of main.
}
