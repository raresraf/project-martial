/**
 * @file 71A.CPP
 * @brief Abbreviates long words to a shorter format.
 *
 * This program implements a word abbreviation logic commonly seen in competitive
 * programming. It processes a series of words, and for each word, it checks
 * its length. If the word is longer than 10 characters, it is abbreviated by
 * taking its first character, followed by the count of characters between the
 * first and last character, and finally the last character. Words with 10 or
 * fewer characters are left unchanged.
 *
 * @note This implementation uses `s.substr(1, s.length() - 2).length()` to
 * calculate the length of the middle part of the string, which is an overly
 * complex way to compute `s.length() - 2`.
 *
 * @algorithm String length check and conditional formatting.
 * @time_complexity O(T * L), where T is the number of test cases and L is the
 * length of the word. Creating a substring may take O(L) time.
 * @space_complexity O(L) to store the longest input word and potentially a substring.
 */
#include <iostream>
// It's good practice to include <string> for std::string operations.
using namespace std;

int main() {
  int amt; // Represents the number of test cases (amount of words).
  cin >> amt;
  
  /**
   * @brief Main processing loop that iterates for each test case.
   * @invariant The loop runs `amt` times to process all words.
   */
  for (int i = 0; i < amt; i++) {
    string s;
    cin >> s;
    
    /**
     * @brief Conditional block to check if the word needs abbreviation.
     * @pre `s` contains a word from the input.
     */
    if (s.length() <= 10) {
      // If the word is 10 characters or shorter, print it as is.
      cout << s << endl;
    } else {
      // If the word's length is greater than 10, abbreviate it.
      // The format is: first character, length-2, last character.
      cout << s[0] << s.substr(1, s.length() - 2).length() << s[s.length() - 1]
           << endl;
    }
  }
}