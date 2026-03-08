/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem (e.g., Codeforces 71A).
 * @details This program reads a number of words and, for each word, checks its length.
 * If the word is longer than 10 characters, it is replaced with a special abbreviation.
 * Otherwise, it is printed as is. The abbreviation consists of the first character,
 * the number of characters between the first and last, and the last character.
 *
 * @algorithm Iterative String Manipulation
 * @complexity
 * - Time: O(N * L), where N is the number of words and L is the maximum length of a word.
 *   The complexity is dominated by the I/O operations for reading N words.
 * - Space: O(L) to store the longest word read.
 */
#include <iostream>
using namespace std;

int main() {
  // The length threshold above which words are abbreviated.
  const int threshold = 10;
  // The total number of words to process.
  int numWords;
  cin >> numWords;

  /**
   * @block Word Processing Loop
   * @logic This do-while loop iterates `numWords` times, processing one word per iteration.
   * The condition `(--numWords)` is a common idiom that decrements and checks in one step.
   */
  do {
    string input;
    cin >> input;
    int inputSize = input.size();

    /**
     * @block Abbreviation Logic
     * @logic Checks if the word's length exceeds the threshold.
     */
    if (inputSize <= threshold) {
      // If the word is not "too long," print it verbatim.
      cout << input << endl;
    } else {
      // If the word is "too long," print its abbreviation.
      // Format: [first letter][number of middle letters][last letter]
      cout << input[0] << inputSize - 2 << input[inputSize - 1] << endl;
    }
  } while (--numWords);

  return 0;
}
