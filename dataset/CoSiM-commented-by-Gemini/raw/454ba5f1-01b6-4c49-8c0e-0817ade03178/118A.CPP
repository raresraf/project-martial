/**
 * @file 118A.CPP
 * @brief Solution to Codeforces problem 118A - String Task.
 *
 * @algorithm
 * The problem requires processing a string according to the following rules:
 * 1. Delete all the vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Before each consonant, insert a character ".".
 * 3. All uppercase consonants should be replaced with their corresponding
 *    lowercase ones.
 *
 * The implementation follows these steps directly:
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. For each character, first convert it to lowercase to handle both
 *    uppercase and lowercase letters uniformly.
 * 4. Check if the lowercase character is a vowel ('a', 'e', 'i', 'o', 'u', 'y').
 * 5. If it is a vowel, ignore it and continue to the next character.
 * 6. If it is a consonant, print a "." followed by the character itself.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the string, because the
 *   code iterates through the string once.
 * - Space Complexity: O(L) to store the input string.
 */
#include <ctype.h>
#include <iostream>
using namespace std;

int main() {
  // a: stores the input string.
  string a;
  // i: loop counter. n: length of the string. j is declared but unused.
  int i, j, n;
  cin >> a;
  n = a.size();
  /**
   * Block Logic: Iterate through each character of the string to apply the rules.
   * Pre-condition: `a` contains the input string.
   * Invariant: Characters up to index `i-1` have been processed and printed if they were consonants.
   */
  for (i = 0; i < n; i++) {
    // Convert the character to lowercase for case-insensitive vowel checking.
    a[i] = tolower(a[i]);
    // Check if the character is a vowel (including 'y').
    if (a[i] == 'a' || a[i] == 'e' || a[i] == 'i' || a[i] == 'o' ||
        a[i] == 'u' || a[i] == 'y')
      // If it's a vowel, skip to the next character.
      continue;
    else
      // If it's a consonant, print a '.' followed by the character.
      cout << "." << a[i];
  }
  return 0;
}
