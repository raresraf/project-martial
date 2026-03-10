/**
 * @file 71A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Way Too Long Words").
 * This program processes a list of words. If a word is longer than 10 characters,
 * it is abbreviated; otherwise, it is printed as is. The abbreviation consists
 * of the first character, the number of characters between the first and last,
 * and the last character.
 *
 * @algorithm The program iterates through each word, checks its length, and applies
 * the abbreviation rule if the length exceeds 10.
 *
 * @time_complexity O(N * L), where N is the number of words and L is the maximum
 * length of a word, dominated by the I/O operations.
 * @space_complexity O(L) to store the longest word read.
 */
#include <iostream>
#include <string>
using namespace std;
using std::getline;
using std::string;

int main() {
  int x; // Number of words to process
  cin >> x;

  /**
   * @block
   * @brief Processes each word according to the abbreviation rule.
   */
  for (int i = 0; i < x; i++) {
    string n;
    cin >> n;
    int tam = n.length();

    // Pre-condition: Check if the word's length is greater than 10.
    if (tam <= 10) {
      // If not, print the word as is.
      cout << n << endl;
    } else {
      // If it is, print the abbreviated version.
      cout << n.at(0) << tam - 2 << n.at(tam - 1) << endl;
    }
  }
}