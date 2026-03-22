/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces (Problem 71A).
 * @details The program reads a number of words and shortens any word that is
 * longer than 10 characters. The abbreviation consists of the first letter,
 * the number of letters in between, and the last letter.
 *
 * @section algorithm
 * Algorithm: Conditional String Manipulation
 * 1. Read the number of test cases, T.
 * 2. Loop T times:
 *    a. Read a word.
 *    b. Check if the word's length is greater than 10.
 *    c. If it is, construct and print the abbreviation.
 *    d. Otherwise, print the word as is.
 *
 * @section complexity
 * - Time Complexity: O(T * L_avg), where T is the number of test cases and
 *   L_avg is the average length of the words. The complexity is linear with
 *   respect to the total input size.
 * - Space Complexity: O(L_max), where L_max is the maximum length of any
 *   single word, required to store the word in memory.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string palabra; // "palabra" is Spanish for "word".
  int t, tam;     // "tam" is short for "tamaño", Spanish for "size".
  cin >> t;

  /**
   * @brief This loop processes each of the 't' test cases.
   */
  while (t--) {
    cin >> palabra;
    tam = palabra.size();

    /**
     * @brief This block implements the core logic of the problem.
     * It checks if the word is "too long" (more than 10 characters) and
     * applies the abbreviation rule accordingly.
     */
    if (palabra.size() > 10) {
      // Constructs the abbreviation: first letter, count of middle letters, last letter.
      cout << palabra.substr(0, 1) << tam - 2
           << palabra.substr(palabra.size() - 1, 1) << endl;
    } else {
      // If the word is not too long, it is printed without modification.
      cout << palabra << endl;
    }
  }
  return 0;
}
