/**
 * @file 71A.CPP
 * @brief This program implements a word abbreviation logic.
 * Words longer than 10 characters are abbreviated to their first letter,
 * followed by the count of characters between the first and last letters,
 * and then the last letter. Words 10 characters or shorter remain unchanged.
 *
 * Algorithm: String length check and formatting.
 * 1. Read the number of words to process.
 * 2. For each word, check its length.
 * 3. If the length is greater than 10, apply the abbreviation rule.
 * 4. Otherwise, print the word as is.
 * Time Complexity: O(L) where L is the total length of all words processed, as each character is visited at most once.
 * Space Complexity: O(Max_Word_Length) to store the current word.
 */

//
// Created by Taewoo Kim on 2/7/2017.
//

#include <iostream>
#include <string>
using namespace std;

/**
 * @brief Main function of the program.
 * Reads words and applies the abbreviation logic, then prints the result.
 * @pre Input consists of an integer `n` followed by `n` words.
 * @post Prints each word, abbreviated if necessary, on a new line.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: 'n' stores the number of words to process. 's' stores the current word.
  int n;
  string s;
  // Functional Utility: Reads the number of words to be processed.
  cin >> n;
  // Block Logic: Iterates 'n' times to process each word.
  // Invariant: At each iteration, a new word is read and processed according to the abbreviation rules.
  for (int i = 0; i < n; i++) {
    // Functional Utility: Reads the current word.
    cin >> s;
    // Block Logic: Checks if the word's length exceeds 10 characters to determine if abbreviation is needed.
    // Pre-condition: `s` contains the current word.
    if (s.size() > 10)
      // Functional Utility: Applies the abbreviation rule: first char + (length - 2) as string + last char.
      s = s[0] + to_string(s.size() - 2) + s[s.size() - 1];
    // Functional Utility: Prints the processed word (abbreviated or original) followed by a newline.
    cout << s << endl;
  }

  return 0;
}
