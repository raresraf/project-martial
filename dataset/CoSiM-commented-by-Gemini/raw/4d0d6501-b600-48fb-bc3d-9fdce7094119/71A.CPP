/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @details This program shortens any given word that is longer than 10
 * characters into an abbreviation. The abbreviation consists of the word's
 * first character, the number of characters between the first and last, and
 * the last character. Words with 10 or fewer characters are not modified.
 *
 * @algorithm
 * 1. Read the number of words `n` to be processed.
 * 2. Loop `n` times to handle each word.
 * 3. In each iteration, read the word into a `std::string`.
 * 4. Check the length of the string.
 * 5. If the length is greater than 10, print the abbreviation.
 * 6. Otherwise, print the original string.
 *
 * @complexity
 * Time Complexity: O(T * L), where T is the number of test cases (words) and L
 * is the length of the longest word, as each word is read and processed.
 * Space Complexity: O(L) to store the longest word.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // n: the number of words to process.
  int n;
  cin >> n;
  
  string s; // Will hold each word as it's read.

  /**
   * @block
   * @brief Main loop to process each of the `n` words.
   * @details The `while(n--)` construct is a common C++ idiom for iterating
   * a fixed number of times.
   */
  while (n--) {
    cin >> s;

    /**
     * @block
     * @brief Core logic for word abbreviation.
     */
    if (s.size() > 10) {
      // If the word is "too long", print its abbreviated form.
      // s[0]: first character
      // s.size() - 2: count of characters between the first and last
      // s[s.size() - 1]: last character
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
    } else {
      // If the word is not too long, print it as is.
      cout << s << endl;
    }
  }

  return 0;
}
