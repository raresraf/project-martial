/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem.
 *
 * @problem Some words are "way too long" and should be abbreviated. If a word has
 * more than 10 characters, it is replaced by an abbreviation. The abbreviation
 * consists of the first character, the number of characters between the first and
 * last, and the last character. Words with 10 or fewer characters remain unchanged.
 * The program applies this rule to a list of 'n' words.
 *
 * @algorithm The program reads an integer 'n', the number of words to process.
 * It then loops 'n' times. In each iteration:
 * 1. It reads a word into a string.
 * 2. It checks the length of the string.
 * 3. If the length is greater than 10, it prints the first character, followed
 *    by the length minus two, followed by the last character.
 * 4. Otherwise, it prints the word as is.
 *
 * @time_complexity O(N*L), where N is the number of words and L is the maximum
 * length of a word. This is dominated by the time taken to read the input.
 * @space_complexity O(L) to store the longest word.
 */
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string> // Included for std::string

using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  // Loop 'n' times to process each word.
  for (int i = 0; i < n; i++) {
    cin >> s;
    // Check if the word is "way too long".
    if (s.size() > 10)
      // If so, print the abbreviation.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << "\n";
    else
      // Otherwise, print the word as is.
      cout << s << "\n";
  }
  return (0);
}
