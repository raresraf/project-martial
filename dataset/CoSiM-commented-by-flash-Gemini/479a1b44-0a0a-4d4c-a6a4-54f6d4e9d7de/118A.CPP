/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem (Codeforces 118A).
 *
 * This program takes an input string and transforms it according to specific rules:
 * 1. All vowels (A, O, Y, E, U, I, case-insensitive) are removed.
 * 2. Before each consonant, a "." character is inserted.
 * 3. All remaining consonants are converted to lowercase.
 * The modified string is then printed to standard output.
 *
 * Algorithm: String traversal and character-by-character processing. It iterates
 * through the input string, applying conditional checks and transformations for
 * each character.
 * Time Complexity: O(L) where L is the length of the input string, as each
 * character is processed a constant number of times.
 * Space Complexity: O(L) to store the input string and potentially the output string
 * (though printed character by character here, effectively O(1) auxiliary space).
 */
#include <iostream>
using namespace std;
/**
 * @brief Main function to transform an input string according to "String Task" rules.
 *
 * Reads a string from standard input, iterates through its characters,
 * removes vowels, prepends a '.' to consonants, and converts consonants
 * to lowercase. The transformed string is printed character by character.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare a string 'input' to store the input word.
  string input;
  // Declare a character 'c' to temporarily store transformed characters.
  char c;
  // Read the input string.
  cin >> input;
  /**
   * @brief Block Logic: Iterates through each character of the input string.
   *
   * Processes each character individually, applying the transformation rules
   * to remove vowels, add '.' before consonants, and convert consonants to lowercase.
   * Pre-condition: 'input' contains the string to be transformed.
   * Invariant: Each character is checked for being a vowel or consonant,
   * and transformed/printed accordingly.
   */
  for (int i = 0; i < input.length(); i++) {
    // Pre-condition: Check if the current character is a vowel (case-insensitive).
    if (input[i] == 'a' || input[i] == 'A' || input[i] == 'e' ||
        input[i] == 'E' || input[i] == 'i' || input[i] == 'I' ||
        input[i] == 'o' || input[i] == 'O' || input[i] == 'u' ||
        input[i] == 'U' || input[i] == 'y' || input[i] == 'Y') {
      // If it's a vowel, skip to the next character (effectively removing it).
      continue;
    } else {
      // If it's a consonant:
      // Print a '.' before the consonant.
      cout << ".";
      // Pre-condition: Check if the consonant is an uppercase letter.
      if (input[i] >= 65 && input[i] <= 90) { // ASCII values for 'A' to 'Z'.
        // Convert the uppercase consonant to lowercase.
        c = input[i] + 32; // Add 32 to ASCII value to convert to lowercase.
        cout << c;        // Print the lowercase consonant.
      } else {
        // If it's already lowercase or another symbol, print as is.
        cout << input[i];
      }
    }
  }
}
