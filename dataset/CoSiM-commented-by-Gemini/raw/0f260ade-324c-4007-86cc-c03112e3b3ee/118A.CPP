/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem (Codeforces 118A).
 * @author competitive-programmer
 *
 * @details This program processes a string according to a specific set of rules:
 * 1. It removes all vowel letters. The vowels are 'A', 'O', 'Y', 'E', 'U', 'I',
 *    and their lowercase counterparts.
 * 2. Before each consonant, it inserts a "." character.
 * 3. It converts all uppercase consonants to their lowercase equivalents.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. For each character, check if it is a vowel.
 * 4. If it is NOT a vowel (i.e., it is a consonant), convert it to lowercase if it's uppercase.
 * 5. Print a "." character followed by the processed lowercase consonant.
 *
 * Time Complexity: O(N), where N is the length of the input string, because we
 * iterate through the string once.
 * Space Complexity: O(N) to store the input string in the character array `a`.
 */
#include <cstring>
#include <iostream>

using namespace std;

int main() {
  int i;
  // `a` is a character array to store the input string.
  char a[101];
  cin >> a;

  /**
   * Block Logic: Iterate through each character of the input string.
   * `strlen(a)` is used to get the length of the C-style string.
   */
  for (i = 0; i < strlen(a); i++) {
    /**
     * Pre-condition: This `if` block checks if the character is NOT a vowel
     * (A, E, I, O, U, Y, and their lowercase versions are considered vowels).
     * If the character is a consonant, the block is executed.
     */
    if (!(a[i] == 'A' || a[i] == 'E' || a[i] == 'I' || a[i] == 'O' ||
          a[i] == 'U' || a[i] == 'a' || a[i] == 'e' || a[i] == 'i' ||
          a[i] == 'o' || a[i] == 'u' || a[i] == 'y' || a[i] == 'Y')) {
      
      /**
       * Logic: If the consonant is an uppercase letter (ASCII 65-90),
       * convert it to lowercase by adding 32 to its ASCII value.
       */
      if (a[i] >= 65 && a[i] <= 90) {
        a[i] += 32;
      }
      
      // Print a "." followed by the processed lowercase consonant.
      cout << "." << a[i];
    }
  }
}
