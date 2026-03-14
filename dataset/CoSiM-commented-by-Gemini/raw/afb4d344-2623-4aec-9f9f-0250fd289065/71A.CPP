/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/71/A
 *
 * This program abbreviates words that are longer than 10 characters.
 */
// CodeForces
// Way too long words
// GNU C++17

#include <iostream>
using namespace std;

int main() {
  int numberOfWords;
  cin >> numberOfWords;
  while (numberOfWords > 0) {
    string currentWord;
    cin >> currentWord;
    /**
     * @brief If the word's length is 10 or less, print it as is.
     */
    if (currentWord.length() <= 10) {
      cout << currentWord << endl;
    } else {
      /**
       * @brief If the word is longer than 10 characters, create an abbreviation.
       * The number of middle characters is calculated with a loop, which is
       * functionally correct but less efficient than simply calculating
       * `currentWord.length() - 2`.
       */
      int count = 0;
      for (int i = 1; i < currentWord.length() - 1; i++) {
        count++;
      }
      cout << currentWord[0] << count << currentWord[currentWord.length() - 1]
           << endl;
    }
    numberOfWords--;
  }
}
