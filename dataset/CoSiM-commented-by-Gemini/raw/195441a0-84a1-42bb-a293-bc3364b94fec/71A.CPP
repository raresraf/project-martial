/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem.
 * @details This program processes a list of words. For each word, if its length is
 * greater than 10, it is replaced by a special abbreviation. Otherwise, the word
 * is printed as is.
 *
 * @section algorithm Algorithm
 * 1. Read an integer `n`, the number of words to be processed.
 * 2. Loop `n` times. In each iteration:
 *    a. Read a word into a string variable.
 *    b. Check the length of the word.
 *    c. If the length is 10 or less, print the word unchanged.
 *    d. If the length is greater than 10, create an abbreviation:
 *       i. Take the first character of the word.
 *       ii. Calculate the number of letters between the first and last characters
 *          (which is `length - 2`).
 *       iii. Convert this number to its string representation.
 *       iv. Take the last character of the word.
 *       v. Concatenate these three parts (first character, number string, last character)
 *          and print the resulting abbreviation.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n * L), where `n` is the number of words and `L` is the
 *   maximum length of a word, dominated by the string I/O operations.
 * - Space Complexity: O(L) to store the longest word.
 */
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * @brief Main function to process words and apply abbreviation logic.
 * @return 0 on successful execution.
 */
int main() {
  int n;
  cin >> n;
  
  // Loop to process each word.
  for (int i = 0; i < n; i++) {
    string word;
    cin >> word;
    
    // Check if the word is "too long".
    if (word.length() <= 10) {
      // If not, print it as is.
      cout << word << endl;
    } else {
      // If it is, create and print the abbreviation.
      // Abbreviation: first letter, count of middle letters, last letter.
      cout << word[0] << to_string(word.length() - 2) << word.back() << endl;
    }
  }
  
  return 0;
}
