/**
 * @file 71A.CPP
 * @brief Abbreviates words that are too long.
 *
 * This program reads a list of words. If a word's length is greater than 10,
 * it is abbreviated by printing its first letter, the number of characters
 * between the first and last letters, and its last letter. Otherwise, the word
 * is printed as is.
 *
 * @b Algorithm: For each word, its length is checked. If it exceeds 10, an
 *    abbreviation is constructed and printed. Otherwise, the original word is printed.
 * @b Time_Complexity: O(N * L), where N is the number of words and L is the
 *    maximum length of a word, due to input reading. The abbreviation logic itself
 *    is O(1) for each word.
 * @b Space_Complexity: O(L) to store the longest input word.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  string inp;
  // Reads the number of words to process.
  scanf("%d", &n);
  // Consumes the rest of the line after reading the integer.
  getline(cin, inp); 
  
  /**
   * @brief This loop processes each word.
   */
  for (int i = 0; i < n; i++) {
    cin >> inp;
    /**
     * @brief Conditional logic for abbreviation.
     * If the word length is greater than 10, it's abbreviated.
     * The abbreviation consists of the first character, the count of characters
     * in between, and the last character.
     */
    if (inp.length() > 10)
      cout << inp[0] << inp.length() - 2 << inp[inp.length() - 1] << endl;
    else
      cout << inp << endl;
  }
  return 0;
}