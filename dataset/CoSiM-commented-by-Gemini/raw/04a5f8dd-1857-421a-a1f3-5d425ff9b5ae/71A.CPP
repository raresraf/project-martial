/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces contest 71.
 *
 * @details
 * This program processes a list of words and abbreviates any word that is longer than 10 characters.
 * The abbreviation consists of the first character of the word, followed by the number of
 * characters between the first and last character, and finally the last character of the word.
 * For example, "localization" becomes "l10n". Words with a length of 10 or less are not modified.
 *
 * The program first reads an integer `n`, which specifies the number of words to process.
 * It then reads `n` words into an array. Finally, it iterates through the array of words,
 * applying the abbreviation logic to each word and printing the result.
 *
 * Algorithm:
 * 1. Read the number of words, `n`.
 * 2. Read `n` words from standard input and store them in an array.
 * 3. Iterate through the array of words from `i = 0` to `n-1`:
 *    a. Get the length of the current word `words[i]`.
 *    b. If the length is greater than 10:
 *       i. Print the first character of the word.
 *       ii. Print the integer value of (length - 2).
 *       iii. Print the last character of the word.
 *    c. If the length is not greater than 10, print the word as is.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum length of a word.
 * This is because reading the N words takes O(N * L) and then iterating and printing takes a similar amount of time.
 *
 * Space Complexity: O(N * L) to store all the words in an array.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  // Read the number of words to be processed.
  cin >> n;
  // Array to store the input words. Note: A fixed-size array might not be safe for all constraints.
  string words[100];

  // Block Logic: Read all 'n' words from the input into the 'words' array.
  for (int i = 0; i < n; i++)
    cin >> words[i];

  // Block Logic: Iterate through each word to apply the abbreviation logic.
  for (int i = 0; i < n; i++) {
    int wordLength = words[i].size();
    // Pre-condition: Check if the word's length exceeds the 10-character limit.
    if (wordLength > 10)
      // Functional Utility: Constructs and prints the abbreviation.
      // It combines the first character, the count of intermediate characters, and the last character.
      cout << words[i][0] << wordLength - 2 << words[i][wordLength - 1] << endl;
    else
      // If the word is not too long, print it without modification.
      cout << words[i] << endl;
  };
  return 0;
}