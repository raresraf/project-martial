/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A, "Way Too Long Words".
 * @details This program abbreviates words that are longer than 10 characters.
 * The abbreviation consists of the first character, the number of characters
 * between the first and last, and the last character. This implementation first
 * reads all words into a vector and then processes them.
 *
 * @algorithm String Abbreviation
 * @complexity
 * Time: O(N * L), where N is the number of words and L is their average length.
 *       The time is dominated by reading the input strings.
 * Space: O(N * L) to store all the words in a vector before processing.
 */
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;

  // A vector to store all the words from the input.
  vector<string> words(n);

  /**
   * @brief First loop: Read all words from the input and store them in the vector.
   */
  for (int i = 0; i < n; i++) {
    cin >> words[i];
  }

  /**
   * @brief Second loop: Iterate through the stored words and apply the abbreviation logic.
   */
  for (int i = 0; i < n; i++) {
    // Check if the word's length is greater than 10.
    if (words[i].size() > 10) {
      // Print the abbreviated version.
      cout << words[i].at(0) << words[i].size() - 2
           << words[i].at(words[i].size() - 1) << endl;
    } else {
      // Print the word as is.
      cout << words[i] << endl;
    }
  }

  return 0; // Indicate successful execution.
}