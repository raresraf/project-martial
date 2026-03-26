/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 * @description This program abbreviates words that are longer than 10 characters.
 *              Words with 10 or fewer characters are printed as is. Longer words
 *              are abbreviated by printing their first character, the count of
 *              characters between the first and last, and their last character.
 *              For example, "localization" becomes "l10n".
 *
 * Algorithm: The program reads an integer `n` specifying the number of words to
 *            process. It then loops `n` times. In each iteration, it reads a word,
 *            checks its length, and applies the abbreviation rule if the length
 *            is greater than 10.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum
 *                  length of a word, due to reading the strings. The processing
 *                  for each word is O(L) for reading and O(1) for abbreviation.
 * Space Complexity: O(L) to store the longest word.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  string word;
  int i, n;

  // Read the number of words to be processed.
  cin >> n;

  /**
   * Block Logic: Loop `n` times to process each word.
   */
  for (i = 0; i < n; i++) {
    cin >> word;
    int wordlen = word.length();
    
    // Pre-condition: Check if the word's length exceeds the threshold of 10.
    if (wordlen <= 10) {
      // If the word is not "too long", print it unchanged.
      cout << word << endl;
    } else {
      // If the word is "too long", print its abbreviation.
      // The format is: first letter, count of middle letters, last letter.
      cout << word[0] << wordlen - 2 << word[wordlen - 1] << endl;
    }
  }
}
