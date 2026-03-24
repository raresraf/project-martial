/**
 * @file 118A.CPP
 * @brief Solution for Codeforces Problem 118A - String Task.
 * @details This program implements a string transformation logic. It processes an input string by deleting all vowels,
 * inserting a '.' character before each consonant, and converting all uppercase consonants to their lowercase counterparts.
 * The letters 'a', 'o', 'y', 'e', 'u', 'i' are considered vowels.
 *
 * @author Andr s Sierra
 * @institute Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 * @algorithm The core logic involves a single pass through the input string. For each character, it first checks if it is a
 * lowercase vowel. If it's not, the character is converted to lowercase. This step handles uppercase vowels and all consonants.
 * A second check is then performed to see if the (now potentially lowercase) character is a vowel. If it's not a vowel,
 * it is a consonant, and it is printed to the console, prefixed with a '.'.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string, as it requires a single loop through the string.
 * Space Complexity: O(N) to store the input string.
 */
// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 */

#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char *argv[]) {

  string t;

  int i = 0;
  cin >> t;

  /**
   * Block Logic: This loop iterates over each character of the string 't' to perform the vowel removal
   * and consonant transformation.
   * Pre-condition: The string 't' has been read from standard input.
   * Invariant: At the beginning of each iteration, `i` is the index of the character to be processed.
   */
  while (i < t.length()) {

    /**
     * Block Logic: This conditional block converts any character that is not a lowercase vowel to its
     * lowercase equivalent. This correctly handles uppercase consonants (e.g., 'B' -> 'b') and also
     * uppercase vowels (e.g., 'A' -> 'a'), preparing them for the vowel check in the next step.
     */
    if (!(t[i] == 'a' || t[i] == 'e' || t[i] == 'i' || t[i] == 'o' ||
          t[i] == 'u' || t[i] == 'y'))
      t[i] = tolower(t[i]);

    /**
     * Block Logic: After potential conversion in the previous step, this block checks if the character
     * is a consonant. Since all uppercase vowels have been converted to lowercase, this check now
     * correctly filters out all vowels. If the character is a consonant, it's printed to the console,
     * prefixed with a '.'.
     */
    if (!(t[i] == 'a' || t[i] == 'e' || t[i] == 'i' || t[i] == 'o' ||
          t[i] == 'u' || t[i] == 'y')) {
      cout << "." << t[i];
    }
    i++;
  }
}
