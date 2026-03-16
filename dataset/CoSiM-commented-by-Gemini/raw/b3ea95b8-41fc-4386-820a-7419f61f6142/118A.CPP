/**
 * @file 118A.CPP
 * @brief Processes a string by removing vowels and adding a '.' before each consonant.
 *
 * This program reads a string, converts it to lowercase, removes all vowels
 * (a, e, i, o, u, y), and inserts a '.' character before each remaining
 * character (consonants). The resulting string is then printed to standard output.
 *
 * @algorithm String Manipulation
 * @details The program iterates through the input string, character by character.
 * It checks if a character is a vowel. If not, it is considered a consonant,
 * and a '.' followed by the consonant is appended to a result vector.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the input string. The program
 * performs a single pass over the string.
 * Space Complexity: O(N), as in the worst-case scenario (a string with no vowels),
 * the new string will be twice the size of the original.
 */

#include <algorithm>
#include <cstring>
#include <ctype.h>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  // Optimizes C++ standard streams for faster I/O.
  ios_base::sync_with_stdio(0);
  string a, ss;
  cin >> a;
  vector<char> v;
  ss = a.size();

  /**
   * @brief Iterate through the string to process each character.
   * For each character, convert it to lowercase and check if it's a vowel.
   */
  for (int i = 0; i < a.size(); ++i) {
    a[i] = tolower(a[i]);
    // Vowels are 'a', 'e', 'i', 'o', 'u', and for this problem, 'y'.
    if (a[i] != 'a' && a[i] != 'e' && a[i] != 'i' && a[i] != 'o' &&
        a[i] != 'u' && a[i] != 'y') {
      // If the character is a consonant, prepend it with a '.'
      v.push_back('.');
      v.push_back((a[i]));
    }
  }

  // Print the resulting vector of characters.
  for (int i = 0; i < v.size(); ++i) {
    cout << v[i];
  }
  return 0;
}
