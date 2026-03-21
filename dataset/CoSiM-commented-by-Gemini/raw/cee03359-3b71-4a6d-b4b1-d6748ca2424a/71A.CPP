/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A, "Way Too Long Words".
 * @details This program processes a series of words. If a word is longer
 *          than 10 characters, it is replaced with a special abbreviation.
 *          Otherwise, it is printed as is.
 *
 * @algorithm
 * The abbreviation is formed by taking the first character of the word,
 * followed by the number of letters between the first and last character,
 * and ending with the last character of the word.
 * For each word, the program first calculates its length. If the length is
 * greater than 10, it prints the abbreviation. Otherwise, it prints the
 * original word.
 *
 * @note This solution uses C-style strings (char arrays) and manual length
 *       calculation, which is less idiomatic and safe than using the C++
 *       `std::string` class.
 *
 * @complexity
 * Time: O(T * L), where T is the number of words and L is the maximum
 *       length of a word, as each character is processed to find the length.
 * Space: O(L) to store the word in a character buffer.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int t;
  cin >> t;
  // Loop to process 't' test cases (words).
  for (int j = 0; j < t; j++) {
    char word[100];
    cin >> word;
    
    int i;
    // Manually calculate the length of the C-style string by finding the null terminator.
    // After this loop, 'i' will hold the length of the word.
    for (i = 0; word[i] != '\0'; i++)
      ;

    // Check if the word's length is greater than 10.
    if (i > 10)
      // If so, print the abbreviation: first char, count of middle chars, last char.
      cout << word[0] << i - 2 << word[i - 1] << endl;
    else
      // Otherwise, print the word as is.
      cout << word << endl;
  }
  return 0;
}
