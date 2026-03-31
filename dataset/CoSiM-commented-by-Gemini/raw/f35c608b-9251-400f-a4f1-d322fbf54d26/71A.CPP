/**
 * @file 71A.CPP
 * @brief A concise C++11 solution for the "Way Too Long Words" problem.
 *
 * This program abbreviates words that are longer than 10 characters. It reads
 * the number of words, `n`, and then processes each word in a loop. If a word's
 * length exceeds 10, it's shortened to its first character, the count of its
 * middle characters, and its last character. The `.back()` string method is
 * used for a clean implementation. Otherwise, the word is printed as is.
 *
 * @author f35c608b-9251-400f-a4f1-d322fbf54d26
 *
 * @algorithm String Size-Checking and Abbreviation
 * @complexity
 * Time: O(N * L), where N is the number of words and L is their average length,
 *       dominated by the string input operations.
 * Space: O(L), to store the longest word at any given time.
 */
#include <iostream>
#include <string> // Included for std::string

using namespace std;

int main() {
  string a;
  int n;
  // Read the number of words to process.
  cin >> n;
  /**
   * @brief Main loop to process each word.
   * The `while(n--)` structure is a common competitive programming idiom.
   */
  while (n--) {
    cin >> a;
    /**
     * @brief Check if the word's length is greater than 10.
     */
    if (a.length() > 10)
      // If so, print the abbreviated form. `a.back()` is a C++11 feature
      // to get the last character. `cout` handles the int-to-string conversion.
      cout << a[0] << (a.length() - 2) << a.back() << "
";
    else
      // Otherwise, print the word in its original form.
      cout << a << "
";
  }
}