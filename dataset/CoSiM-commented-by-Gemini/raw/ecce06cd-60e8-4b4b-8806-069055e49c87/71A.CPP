/**
 * @file 71A.CPP
 * @brief Abbreviates words that are "too long".
 *
 * This program implements a solution to a common coding problem where words longer
 * than 10 characters are abbreviated. The abbreviation consists of the first
 * letter, the number of letters between the first and last, and the last letter.
 *
 * @b Algorithm: The program iterates through a list of words. For each word, it
 *    checks its length. If the length is greater than 10, it constructs the
 *    abbreviated form. Otherwise, it leaves the word unchanged.
 * @b Time_Complexity: O(N * L), where N is the number of words and L is the maximum
 *    word length, dominated by I/O and string operations.
 * @b Space_Complexity: O(N * L) to store the words. Note that a C-style
 *    variable-length array is used, which is a non-standard C++ extension.
 */
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

int main() {

  // Read the number of words to process.
  int n;
  cin >> n;

  // Declare a variable-length array to store the words.
  // Note: A std::vector<string> would be the standard C++ approach.
  string palabras[n];

  // Read n words into the array.
  for (int i = 0; i < n; i++) {
    string word;
    cin >> word;
    palabras[i] = word;
  }

  // Process each word.
  for (int j = 0; j < n; j++) {
    // Pre-condition: Check if the word is longer than 10 characters.
    if (palabras[j].length() > 10) {

      // The number of letters between the first and last is simply length - 2.
      // The original code uses a loop, which is functionally correct but verbose.
      int count = palabras[j].length() - 2;

      // Convert the count to its string representation.
      string result;
      stringstream sstm;
      sstm << count;
      result = sstm.str();

      // Replace the middle characters of the word with the string representation of the count.
      // For example, "localization" becomes "l10n".
      palabras[j] = palabras[j].replace(palabras[j].begin() + 1,
                                        palabras[j].end() - 1, result);
      cout << palabras[j] << endl;
    } else {
      // If the word is not too long, print it as is.
      cout << palabras[j] << endl;
    }
  }

  return 0;
}
