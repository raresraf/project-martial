/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 * @description This program implements a string abbreviation rule. If a word's length
 *              is greater than 10, it is shortened to a format consisting of its
 *              first letter, the number of letters between the first and last, and
 *              its last letter. Otherwise, the word is printed as is.
 *
 * Algorithm: The program first reads the number of words to process. It then enters
 *            a loop that runs for that many iterations. Inside the loop, each word
 *            is read, its length is checked, and based on the length, either the
 *            original word or its abbreviated form is printed to standard output.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum
 *                  word length, dominated by the I/O operations.
 * Space Complexity: O(L) to store the longest word.
 */
// CodeForces
// Problem 71A [Way Too Long Words]

#include <iostream>
#include <string>
using namespace std;

int main() {
  string input;

  unsigned int test_cases = 0;
  // Read the number of words to process.
  cin >> test_cases;

  /**
   * Block Logic: Loop through all the test cases.
   * The `while(test_cases--)` construct is a common idiom for this.
   */
  while (test_cases--) {
    cin >> input;

    // Pre-condition: Check if the word length is greater than 10.
    if (input.length() > 10) {
      // If so, print the abbreviated version.
      cout << input[0] << input.length() - 2 << input[input.length() - 1]
           << endl;
    } else {
      // Otherwise, print the word as is.
      cout << input << endl;
    }
  }
  return 0;
}
