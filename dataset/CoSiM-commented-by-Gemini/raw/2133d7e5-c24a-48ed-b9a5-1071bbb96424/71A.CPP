/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A, "Way Too Long Words".
 * @details This program processes a list of words. For any word that is longer
 * than 10 characters, it creates an abbreviation consisting of the first letter,
 * the count of letters in between, and the last letter. Words with 10 or fewer
 * characters are left unchanged.
 *
 * @algorithm Iterative processing with conditional string manipulation.
 * @description The program first reads the number of words to process. It then
 * loops that many times, reading one word per iteration. Inside the loop, it
 * checks the word's length and prints either the original word or its
 * abbreviation based on the problem's rules.
 * @complexity
 * - Time: O(N), where N is the number of words, as each word is processed once.
 *   String operations inside the loop are effectively O(1) as word length is small.
 * - Space: O(L), where L is the maximum length of a word, to store the input string.
 */
#include <iostream>
#include <string> // Added for std::string

int main() {
  int n;
  std::cin >> n;
  std::string s;

  /**
   * @brief This loop iterates 'n' times to process each word.
   */
  while (n--) {
    std::cin >> s;
    int size = s.size();

    /**
     * @brief This block checks if the word's length is greater than 10.
     * If it is, the word is abbreviated; otherwise, it is printed as is.
     * The abbreviation format is: first character, number of middle characters, last character.
     */
    if (size > 10) {
      std::cout << s[0] << size - 2 << s[size - 1] << '\n';
    } else {
      std::cout << s << '\n';
    }
  }
}
