/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem (Codeforces 71A).
 *
 * This program processes a list of words. For each word, if its length
 * exceeds 10 characters, it prints an abbreviated version consisting of
 * the first character, the count of characters between the first and last
 * (as a number), and the last character. Otherwise, the original word is printed.
 *
 * Algorithm: String manipulation and conditional logic. Each string's length
 * is checked, and either abbreviated or printed as is.
 * Time Complexity: O(N * L) where N is the number of words and L is the average
 * length of a word. String operations like `size()`, `to_string()`, and `[]`
 * are generally efficient.
 * Space Complexity: O(L) to store the current word.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function to process words and apply abbreviation logic.
 *
 * Reads an integer 'n' specifying the number of words to process.
 * For each word, it checks its length and either prints an abbreviated
 * version or the original word.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare an integer 'n' for the number of words to process.
  int n;
  // Read 'n' from standard input.
  cin >> n;
  // Declare a string 's' to store each word.
  string s;
  /**
   * @brief Block Logic: Processes 'n' words.
   *
   * Iterates 'n' times, reading each word and applying the abbreviation logic.
   * Pre-condition: 'n' specifies the total number of words.
   * Invariant: Each word is processed according to its length, and the result is printed.
   */
  while (n--) {
    // Read the current word into string 's'.
    cin >> s;
    // Pre-condition: 's' contains the current word.
    // Block Logic: Checks if the length of the word is greater than 10.
    if (s.size() > 10)
      // If true, print the abbreviated form: first character + (length - 2) + last character.
      cout << s[0] + to_string(s.size() - 2) + s.back() << endl;
    else
      // If false, print the original word.
      cout << s << endl;
  }
  return 0;
}