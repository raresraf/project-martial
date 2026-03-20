/**
 * @file 71A.CPP
 * @brief Solution attempt for Codeforces problem 71A - Way Too Long Words.
 *        This program processes a list of words, aiming to abbreviate those that are
 *        "too long" according to a specific interpretation.
 * Algorithm:
 *   1. Reads an integer 't', indicating the number of words to process.
 *   2. Initializes a vector of strings `use_case` to store abbreviated words.
 *   3. For each of the 't' words:
 *      a. Reads a word (string `h`).
 *      b. If the length of `h` is greater than or equal to 10:
 *         i. Extracts the first character.
 *         ii. Converts the string length to a string.
 *         iii. Extracts the last character.
 *         iv. Concatenates these parts in the format: first_char + string_length_as_string + last_char.
 *             (e.g., "localization" (length 12) becomes "l12n", not "l10n").
 *         v. Stores this abbreviated string in `use_case`.
 *   4. After processing all words, iterates through `use_case` and prints each stored string.
 * Deviations/Observations:
 *   - The problem typically defines "too long" as strictly greater than 10 characters,
 *     whereas this code uses `>= 10`.
 *   - The standard abbreviation format is first_char + (length-2) + last_char. This code
 *     uses first_char + length_as_string + last_char.
 *   - Words shorter than 10 characters are not printed by this implementation.
 * Time Complexity: O(T * L), where T is the number of test cases and L is the average length
 *                  of the words, as it iterates through each character for length and substring operations.
 * Space Complexity: O(T * L_max) in the worst case to store `use_case`, where L_max is the max word length.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * @brief Main function to process words and apply a specific abbreviation logic.
   *
   * Reads a series of words and, for those meeting a length threshold (>= 10 characters),
   * constructs an abbreviated version (first_char + string_length + last_char).
   * It then prints all such abbreviated words. Words shorter than the threshold
   * are not printed by this implementation.
   * @return int Returns 0 upon successful execution.
   */
  int t;
  cin >> t; // Read the number of words to process.
  vector<string> use_case; // Vector to store the abbreviated words.

  // Block Logic: Loops 't' times to process each input word.
  // Pre-condition: 't' specifies the total number of words.
  // Invariant: For each iteration, a word is read and potentially abbreviated.
  for (int i = 0; i < t; i++) {
    string h, c1, c3;
    cin >> h; // Read the current word.
    c1 = h[0]; // Get the first character of the word.
    int j = h.length(); // Get the length of the word.
    std::string c2 = std::to_string(j); // Convert word length to string.
    c3 = h[j - 1]; // Get the last character of the word.
    c2.append(c3); // Concatenate length string with the last character.
    c1.append(c2); // Concatenate first character with the combined length and last character string.

    // Block Logic: Checks if the word's length is 10 or more for abbreviation.
    // Note: The problem typically specifies '> 10' for abbreviation, not '>= 10'.
    // Also, words with length <= 9 are not added to 'use_case' and thus not printed.
    if (h.length() >= 10) {
      use_case.push_back(c1); // Store the custom-abbreviated word.
    }
  }
  // Block Logic: Prints all the stored abbreviated words, each on a new line.
  // Pre-condition: 'use_case' contains all words that met the abbreviation criteria.
  for (int i = 0; i < use_case.size(); i++) {
    cout << use_case[i] << endl;
  }
  return 0;
}
