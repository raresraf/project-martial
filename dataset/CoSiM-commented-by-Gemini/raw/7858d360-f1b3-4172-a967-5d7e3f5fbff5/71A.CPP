/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from competitive programming.
 *
 * This program reads an integer `n` followed by `n` words. For each word,
 * if its length is greater than 10 characters, it's abbreviated. Otherwise,
 * it's printed as is. The abbreviation consists of the first character,
 * the number of characters between the first and last, and the last character.
 *
 * @algorithm
 * 1. Read the number of test cases, `n`.
 * 2. Loop `n` times to process each word.
 *    a. Read the word into a string `s`.
 *    b. Calculate the length of the string, `len`.
 *    c. Check if `len` is less than or equal to 10.
 *       - If true, print the string `s` directly.
 *    d. Else (if `len` is greater than 10):
 *       - Print the first character, `s[0]`.
 *       - Print the integer value `len - 2`.
 *       - Print the last character, `s[len - 1]`.
 *
 * @complexity
 * Time Complexity: O(n * L), where `n` is the number of words and `L` is the
 * maximum length of a word. The complexity is dominated by the input/output
 * operations.
 * Space Complexity: O(L) to store the longest word.
 */
#include <iostream>
#include <string> // It's good practice to include the string header for std::string

using namespace std;

int main() {
  // `s` will store the word for each test case.
  string s;
  // `n` is the number of words to process.
  int n;
  cin >> n;
  
  int len = 0; // Variable to store the length of the string.

  /**
   * @brief Main loop to process `n` words.
   */
  for (int i = 0; i < n; i++) {
    cin >> s;
    len = s.length();

    // Check if the word needs to be abbreviated.
    if (len <= 10) {
      // If not, print it as is.
      cout << s << endl;
    } else {
      // If it's too long, print the abbreviation.
      cout << s[0] << len - 2 << s[len - 1] << endl;
    }
  }

  return 0;
}
