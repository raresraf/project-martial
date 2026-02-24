/**
 * @file raw/454ba5f1-01b6-4c49-8c0e-0817ade03178/118A.CPP
 * @brief Solution to Codeforces problem 118A - String Task.
 *
 * @algorithm
 * The problem requires processing a string based on a set of rules:
 * 1. Convert all uppercase letters to lowercase.
 * 2. Delete all vowels. The vowels are 'a', 'e', 'i', 'o', 'u', and 'y'.
 * 3. Before each consonant, insert a '.' character.
 *
 * This implementation processes the string character by character and builds
 * the output string live by printing to standard output.
 *
 * The algorithm is as follows:
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. For each character, convert it to its lowercase equivalent.
 * 4. Check if the lowercase character is a vowel ('a', 'e', 'i', 'o', 'u', 'y').
 * 5. If it is a vowel, ignore it and continue to the next character.
 * 6. If it is a consonant, print a '.' character followed by the consonant itself.
 *
 * This approach effectively transforms the string in a single pass.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the input string, as it
 *   involves a single pass through the string.
 * - Space Complexity: O(L) to store the input string. The output is printed
 *   directly, so auxiliary space is O(1).
 */
#include <ctype.h>
#include <iostream>
using namespace std;

int main() {
  string a;
  int i, j, n; // j is declared but unused.
  cin >> a;
  n = a.size();

  /**
   * @brief Iterate through the string to process each character.
   */
  for (i = 0; i < n; i++) {
    // Convert the character to lowercase.
    a[i] = tolower(a[i]);

    // Check if the character is a vowel. The problem considers 'y' a vowel.
    if (a[i] == 'a' || a[i] == 'e' || a[i] == 'i' || a[i] == 'o' ||
        a[i] == 'u' || a[i] == 'y') {
      // If it's a vowel, skip it.
      continue;
    } else {
      // If it's a consonant, print a '.' followed by the character.
      cout << "." << a[i];
    }
  }
  
  return 0;
}
