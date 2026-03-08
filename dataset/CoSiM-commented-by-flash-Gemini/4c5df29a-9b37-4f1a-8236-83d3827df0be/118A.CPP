/**
 * @file 118A.CPP
 * @brief Processes an input word by converting to lowercase, removing vowels, and formatting consonants with preceding dots.
 *
 * Algorithm: Iterative character processing for case normalization, vowel filtering, and output formatting.
 * Time Complexity: O(L) where L is the length of the input string, due to a single pass and string length calculation.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, including string functions.
using namespace std;

/**
 * @brief Main function to process the input word as per problem rules.
 * @pre Input consists of a single word (string).
 * @post Prints the processed string with lowercase consonants, each preceded by a dot.
 */
int main() {
  int i;         // Loop counter.
  char s[100];   // Character array to store the input word (max length 99 + null terminator).
  // Functional Utility: Reads the input word into the character array 's'.
  cin >> s;

  /**
   * Block Logic: Iterates through each character of the input string to apply transformations.
   * Invariant: For each character, it first ensures it's lowercase, then checks if it's a consonant to print it with a dot.
   */
  for (i = 0; i < strlen(s); i++) {
    /**
     * Block Logic: Converts uppercase letters to lowercase.
     * Pre-condition: 's[i]' is the current character.
     * Invariant: If 's[i]' is an uppercase letter (ASCII 65-90), it is converted to its lowercase equivalent (subtracting 65 and adding 97).
     */
    if (s[i] >= 65 && s[i] <= 90) { // Checks if the character is an uppercase letter.
      s[i] = 97 + s[i] - 65;        // Converts to lowercase using ASCII offset.
    }

    /**
     * Block Logic: Checks if the character is a consonant and prints it with a dot.
     * Pre-condition: 's[i]' is the current (now lowercase) character.
     * Invariant: If 's[i]' is not one of the specified vowels ('a', 'e', 'i', 'o', 'u', 'y'), it is printed.
     */
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y') {
      printf(".%c", s[i]); // Prints a dot followed by the consonant.
    }
  }
  // Indicates successful program execution.
  return 0;
}