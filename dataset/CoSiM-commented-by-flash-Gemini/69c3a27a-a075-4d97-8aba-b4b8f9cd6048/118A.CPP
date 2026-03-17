/**
 * @69c3a27a-a075-4d97-8aba-b4b8f9cd6048/118A.CPP
 * @brief Problem: StringTask.
 *        Transforms an input string by converting all characters to lowercase,
 *        removing all vowels (a, o, u, e, i, y), and inserting a '.' character
 *        before each consonant.
 * Algorithm: Character filtering and string concatenation.
 *            1. Define a helper function to check if a character is a vowel (case-insensitive).
 *            2. Read the input string.
 *            3. Iterate through each character of the input string.
 *            4. For each character, convert it to lowercase.
 *            5. If the character is not a vowel, prepend a '.' and append the lowercase character
 *               to an output string.
 *            6. Print the resulting transformed string.
 * Time Complexity: O(L) where L is the length of the input string.
 *                  Each character is processed a constant number of times (lowercase conversion, vowel check, concatenation).
 * Space Complexity: O(L) for storing the input and output strings.
 */

// CodeForces
// 118A - StringTask

#include <iostream> // Functional Utility: Provides cin and cout for input/output.
#include <string>   // Functional Utility: Provides string class for string manipulation.
#include <cctype>   // Functional Utility: Provides tolower() for character case conversion.

using namespace std;

/**
 * @brief Checks if a given character is a vowel (a, o, u, e, i, y), case-insensitive.
 * @param c The character to check.
 * @return True if the character is a vowel, false otherwise.
 * Pre-condition: 'c' is a valid character.
 * Post-condition: Returns boolean indicating vowel status.
 */
bool isVowel(char c) {
  c = (char)tolower(c); // Functional Utility: Converts character to lowercase for consistent comparison.

  // Block Logic: Checks if the lowercase character matches any of the defined vowels.
  if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y')
    return true; // Functional Utility: Indicates the character is a vowel.
  else
    return false; // Functional Utility: Indicates the character is not a vowel.
}

int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string word;
  // Block Logic: Reads the input word from standard input.
  cin >> word;

  string output(""); // Invariant: 'output' accumulates the transformed string.

  // Block Logic: Iterates through each character of the input word.
  // Pre-condition: 'word' contains the string to be processed.
  // Invariant: 'output' is built character by character based on vowel/consonant checks.
  for (unsigned int i = 0; i < word.size(); i++) {
    // Block Logic: Checks if the current character is a consonant.
    // Pre-condition: 'word[i]' is the character to be evaluated.
    // Post-condition: If 'word[i]' is a consonant, it is added to 'output' prefixed with a '.'.
    if (!isVowel(word[i])) {
      // Functional Utility: Appends a '.' and the lowercase version of the consonant to the output string.
      output = output + "." + (char)tolower(word[i]);
    }
  }

  // Functional Utility: Prints the final transformed string.
  cout << output;

  return 0; // Functional Utility: Indicates successful program execution.
}
