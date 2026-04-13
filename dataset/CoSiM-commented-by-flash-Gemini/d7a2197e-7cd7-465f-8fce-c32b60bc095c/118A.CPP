/**
 * @file 118A.CPP
 * @brief This program transforms an input string according to specific rules:
 * 1. All vowels (case-insensitive: A, O, Y, E, U, I) are removed.
 * 2. All remaining characters are converted to lowercase.
 * 3. A '.' character is inserted before each remaining consonant.
 * Algorithm: Iterative string processing. First pass removes vowels, second converts to lowercase,
 * third constructs the final string with dot prefixes.
 * Time Complexity: O(N^2) in the worst case for vowel removal due to `string::erase` in a loop,
 * where N is the length of the input string.
 * Space Complexity: O(N) for storing the input and output strings.
 */
#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for std::string and its methods.
#include <cctype>   // Required for isupper and tolower functions.

using namespace std; // Brings all names from the std namespace into the current scope.

int main() {
  string word1;  // Stores the input string, which will be modified in place.
  string word2;  // Stores the final transformed output string.
  // Reads the input string from standard input.
  cin >> word1;

  /**
   * Block Logic: Iterates through the string `word1` to remove all vowel characters.
   * Both lowercase and uppercase vowels ('a', 'o', 'y', 'e', 'u', 'i') are removed.
   * When a character is erased, the loop index `i` is decremented to re-check the
   * character that shifts into the current position.
   * Invariant: `word1` progressively becomes shorter, containing only consonants.
   */
  for (int i = 0; i < word1.length(); i++) {
    // Checks if the current character is a vowel (case-insensitive).
    if (word1[i] == 'a' || word1[i] == 'A' || word1[i] == 'o' ||
        word1[i] == 'O' || word1[i] == 'y' || word1[i] == 'Y' ||
        word1[i] == 'E' || word1[i] == 'e' || word1[i] == 'u' ||
        word1[i] == 'U' || word1[i] == 'i' || word1[i] == 'I') {
      word1.erase(i, 1); // Erases the vowel character from the string.
      i--;               // Decrements loop index to re-evaluate the character that shifted into this position.
    }
  }

  /**
   * Block Logic: Converts all remaining characters in `word1` to lowercase.
   * This ensures that all consonants are in lowercase before prefixing.
   * Invariant: All characters in `word1` are now lowercase consonants.
   */
  for (int i = 0; i < word1.length(); ++i) {
    if (isupper(word1[i]))     // Checks if the character is an uppercase letter.
      word1[i] = tolower(word1[i]); // Converts it to lowercase.
  }

  /**
   * Block Logic: Constructs the final output string `word2` by prefixing each consonant with a dot.
   * It iterates through the processed `word1` (lowercase consonants) and appends
   * a "." followed by the character to `word2`.
   * Invariant: `word2` is built with ".consonant" sequence for each character in `word1`.
   */
  for (int j = 0; j < word1.length(); j++) {
    word2 = word2 + "." + word1[j]; // Prefixes with '.' and appends the character.
  }
  // Prints the final transformed string `word2` to standard output, followed by a newline.
  cout << word2 << endl;

  // Standard program exit.
  return 0;
}