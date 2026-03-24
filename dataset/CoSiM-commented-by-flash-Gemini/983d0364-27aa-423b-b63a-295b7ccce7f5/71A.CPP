/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem: "Way Too Long Words".
 *
 * Implements a word abbreviation logic: if a word's length exceeds 10 characters,
 * it is replaced by its first character, followed by the count of intermediate characters,
 * and then its last character. Shorter words remain unchanged.
 *
 * Algorithm: Iterative string processing with conditional formatting.
 * Time Complexity: O(X * L), where X is the number of words and L is the average length of words.
 */
#include <iostream>
#include <string>
using namespace std;
using std::getline;
using std::string;

/**
 * @brief Main function to process a list of words and apply abbreviation rules.
 * Reads the number of words 'x', then for each word, determines if it needs
 * to be abbreviated based on its length and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  /**
   * Declares 'x', the total number of words to be processed.
   * Reads the value of 'x' from standard input.
   * Pre-condition: Standard input provides an integer 'x'.
   *
   * Block Logic: Iterates 'x' times, processing one word in each iteration.
   * Invariant: Each iteration handles the abbreviation logic for a single word.
   * Pre-condition: 'x' is a non-negative integer.
   */
  int x;
  cin >> x;
  for (int i = 0; i < x; i++) {
    /**
     * Declares 'n' to store the current word and reads it from standard input.
     * 'tam' stores the length of the current word.
     * Pre-condition: Standard input provides a string.
     */
    string n;
    cin >> n;
    int tam = n.length();
    /**
     * Block Logic: Applies the abbreviation rule based on the word's length.
     * If the length is 10 or less, prints the word as is.
     * If the length is greater than 10, prints the first character,
     * followed by (length - 2), and then the last character.
     */
    if (tam <= 10) {
      cout << n << endl;
    } else {
      cout << n.at(0) << tam - 2 << n.at(tam - 1) << endl;
    }
  }
}