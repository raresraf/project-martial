/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 *
 * @details
 * This program processes input strings according to a set of rules: delete vowels,
 * insert a '.' before each consonant, and convert consonants to lowercase. It is
 * structured to handle multiple input strings, one per line.
 *
 * Algorithm:
 * 1. Loop to read each input string until the end of input.
 * 2. For each string, iterate through its characters.
 * 3. For each character, check if it is a lowercase consonant. If so, print a '.'
 *    followed by the character.
 * 4. Check if it is an uppercase consonant. If so, print a '.' followed by the
 *    character's lowercase version.
 * 5. Vowels are implicitly ignored by this logic.
 * 6. Print a newline after processing each string.
 *
 * This implementation handles uppercase and lowercase consonants in separate
 * conditional blocks, which is more verbose than converting all characters to
 * lowercase first.
 *
 * Time Complexity: O(L) for each input string of length L.
 * Space Complexity: O(L) to store the input string.
 */
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int i, j, k, l; // Note: j and k are declared but never used.
  string s;
  // The while loop allows processing multiple test cases (one string per line).
  while (cin >> s) {
    l = s.length();
    for (i = 0; i < l; i++) {
      // Check if the character is a lowercase consonant.
      if (s[i] >= 'a' && s[i] <= 'z' && s[i] != 'a' && s[i] != 'e' &&
          s[i] != 'i' && s[i] != 'o' && s[i] != 'y' && s[i] != 'u') {
        cout << ".";
        cout << s[i];
      } 
      // Check if the character is an uppercase consonant.
      else if (s[i] >= 'A' && s[i] <= 'Z' && s[i] != 'A' && s[i] != 'E' &&
                 s[i] != 'I' && s[i] != 'O' && s[i] != 'Y' && s[i] != 'U') {
        cout << ".";
        // Convert the uppercase consonant to lowercase before printing.
        cout << (char)tolower(s[i]);
      }
      // Vowels are implicitly ignored as they don't satisfy either condition.
    }
    cout << endl;
  }
  return 0;
}
