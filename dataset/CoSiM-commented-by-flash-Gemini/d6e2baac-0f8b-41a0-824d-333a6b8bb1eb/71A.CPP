/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem.
 *
 * This program processes a list of words. For each word, if its length exceeds 10 characters,
 * it is abbreviated to its first character, followed by the count of characters between
 * the first and last characters, and then its last character. Otherwise, the word
 * remains unchanged.
 *
 * Algorithm: String length check and abbreviation.
 * Time Complexity: O(N * L) where N is the number of words and L is the maximum length of a word.
 *                  This is because each character of each word is processed at most once.
 * Space Complexity: O(N * L) to store the words in a vector.
 */
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * @brief Main function to process words and apply abbreviation rules.
 *
 * Functional Utility: Reads a list of words, checks their lengths, and prints
 * either the original word or its abbreviated form.
 */
int main() {
  int n;
  cin >> n; // Read the number of words to process.
  vector<string> words(n); // Declare a vector to store the words.

  /**
   * Block Logic: Reads 'n' words from standard input and stores them in a vector.
   * Pre-condition: 'n' has been read, indicating the number of words to follow.
   * Invariant: After each iteration, one word is read and stored.
   */
  for (int i = 0; i < n; i++) {
    cin >> words[i];
  }

  /**
   * Block Logic: Iterates through each word in the vector and applies the abbreviation rule.
   * For words longer than 10 characters, it prints the abbreviated form (first char + length-2 + last char).
   * Otherwise, it prints the original word.
   * Invariant: Each word is processed exactly once, and its appropriate form is printed.
   */
  for (int i = 0; i < n; i++) {
    if (words[i].size() > 10) {
      cout << words[i].at(0) << words[i].size() - 2
           << words[i].at(words[i].size() - 1) << endl;
    } else {
      cout << words[i] << endl;
    }
  }
}