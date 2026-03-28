/**
 * @file 71A.CPP
 * @brief A clean and efficient solution for the "Way Too Long Words" problem.
 *
 * This program abbreviates words that are longer than 10 characters. It processes
 * each word as it is read from standard input, making it memory-efficient.
 *
 * @b Algorithm: The program iterates `n` times. In each iteration, it reads a word,
 *    checks its length against a threshold, and prints either the original word or
 *    its abbreviated form. The abbreviation consists of the first letter, the count
 *    of middle letters, and the last letter.
 * @b Time_Complexity: O(N * L), where N is the number of words and L is their
 *    maximum length. The complexity is dominated by the I/O operations.
 * @b Space_Complexity: O(L), as only one word is stored in memory at a time.
 */
#include <iostream>
#include <string> // Added for completeness, although often included by <iostream>

using namespace std;

int main() {
  const int threshold = 10;
  int numWords;
  cin >> numWords;
  
  /**
   * @brief Main loop to process each word from the input stream.
   * A do-while loop is used here, processing `numWords` entries.
   */
  do {
    string input;
    cin >> input;
    int inputSize = input.size();

    // Check if the word length exceeds the threshold.
    if (inputSize <= threshold) {
      cout << input << endl;
    } else {
      // If the word is too long, print its abbreviation.
      cout << input[0] << inputSize - 2 << input[inputSize - 1] << endl;
    }
  } while (--numWords);

  return 0;
}