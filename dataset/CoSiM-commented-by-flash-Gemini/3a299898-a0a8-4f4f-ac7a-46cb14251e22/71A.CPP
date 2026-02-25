/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A - Way Too Long Words.
 *
 * This program takes a list of words and abbreviates any word that is
 * strictly longer than 10 characters. The abbreviation consists of the
 * first letter of the word, followed by the number of characters between
 * the first and last letters, and then the last letter of the word.
 * Words 10 characters or shorter are printed as is.
 *
 * Algorithm: String length check and conditional formatting. For each word,
 * its length is checked. If it exceeds 10, it's abbreviated; otherwise, it's printed whole.
 * Time Complexity: O(N * L), where N is the number of words and L is the average
 *                  length of a word (as string operations like length() are O(L)).
 * Space Complexity: O(L_max) for storing the longest word.
 */

#include <iostream> // Required for standard input/output operations (cin, cout).
#include <string>   // Required for std::string.

using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function to execute the Way Too Long Words problem logic.
   *
   * Optimizes C++ standard stream I/O for faster execution.
   * Reads an integer 'n' representing the number of words to process.
   * Then, it enters a loop to read 'n' words one by one. For each word,
   * it applies the abbreviation logic and prints the result to standard output.
   */
  // Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout.
  // This optimizes high-throughput I/O often required for competitive programming.
  ios::sync_with_stdio(0);
  cin.tie(0);

  // Declare 'n' to store the number of words.
  int n;
  // Read the total number of words to be processed.
  cin >> n;

  // Block Logic: Process each word.
  // Invariant: The loop iterates 'n' times, processing one word in each iteration.
  for (int i = 0; i < n; i++) {
    // Declare 's' to store the current word.
    string s;
    // Read the current word.
    cin >> s;

    // Get the length of the current word.
    int len = s.length();

    // Conditional Logic: Check if the word needs abbreviation.
    // Pre-condition: 'len' holds the length of the string 's'.
    if (len > 10) {
      // Print the abbreviated word: first character, count of middle characters, last character.
      cout << s[0] << len - 2 << s[len - 1] << endl;
    } else {
      // If the word is 10 characters or shorter, print it as is, followed by a newline.
      cout << s << endl;
    }
  }

  return 0; // Indicate successful program execution.
}
