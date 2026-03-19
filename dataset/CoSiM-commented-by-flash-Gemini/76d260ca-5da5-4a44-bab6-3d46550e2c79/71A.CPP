/**
 * @file 71A.CPP
 * @brief Solution to Codeforces Problem 71A: "Way Too Long Words".
 *
 * This problem involves abbreviating words that are "too long". A word is considered
 * too long if its length is strictly greater than 10 characters. For such words,
 * the abbreviation consists of the first letter, followed by the number of
 * characters between the first and last letters, and then the last letter.
 * Words that are not too long remain unchanged.
 *
 * Algorithm: String length check and conditional abbreviation.
 * For each word, the algorithm checks its length. If the length is greater than 10,
 * it constructs the abbreviation using the first character, the length minus 2
 * (representing the characters between the first and last), and the last character.
 * Otherwise, the original word is printed.
 *
 * Time Complexity: O(L_avg * N), where N is the number of words and L_avg is the average
 *                  length of the words. String length calculation and character access
 *                  are typically O(L) for a word of length L.
 * Space Complexity: O(L_max) for storing the longest word.
 */
#include <bits/stdc++.h> // Includes various standard libraries, such as iostream and string.

#define ll long long int // Defines 'll' as a shorthand for 'long long int'. (Unused in this specific problem).

#define mod 1000000007 // Defines 'mod' as a constant for modulo operations. (Unused in this specific problem).

using namespace std; // Uses the standard namespace to avoid prefixing std::

/**
 * @brief Main function to process a list of words and print their abbreviated
 *        or original forms based on length.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
  int n; // Stores the number of words to process.
  cin >> n; // Reads the number of words.

  /**
   * Block Logic: Processes each word one by one.
   * Pre-condition: 'n' specifies the number of words to evaluate.
   * Invariant: After each iteration, one word has been processed and printed.
   */
  while (n--) {
    string s; // Stores the current word.
    cin >> s; // Reads the current word.

    // Block Logic: Checks if the word's length requires abbreviation.
    if (s.length() > 10) {
      // Prints the abbreviated form: first character, (length - 2), last character.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      // Prints the original word if its length is 10 or less.
      cout << s << endl;
    }
  }
  return 0; // Indicates successful program execution.
}