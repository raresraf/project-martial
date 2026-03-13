/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 * @details The program transforms an input string by removing all vowels,
 * inserting a "." character before each consonant, and converting all
 * uppercase consonants to their lowercase equivalents.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Perform a first pass to convert the entire string to lowercase. This
 *    implementation does so by checking the ASCII range for uppercase letters
 *    and adding 32.
 * 3. Perform a second pass through the now-lowercase string.
 * 4. In the second pass, check if a character is a vowel ('a', 'e', 'i', 'o', 'u', 'y').
 * 5. If the character is not a vowel (i.e., it's a consonant), print a '.'
 *    followed by the character.
 * 6. Finally, print a newline character.
 *
 * Time Complexity: O(N), where N is the length of the string, due to two separate passes.
 * Space Complexity: O(N), to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  cin >> s;

  /**
   * Block Logic: First pass to convert the entire string to lowercase.
   * Invariant: After this loop, the string 's' contains no uppercase characters.
   */
  for (int i = 0; i < s.size(); ++i) {
    // Check if the character is an uppercase letter using its ASCII value.
    if (s[i] >= 65 && s[i] <= 90) {
      // Convert to lowercase by adding 32 to the ASCII value.
      s[i] = s[i] + 32;
    }
  }

  /**
   * Block Logic: Second pass to process vowels and consonants.
   * Iterates through the lowercase string, printing a '.' before each consonant.
   */
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y') {
      cout << "." << s[i];
    }
  }

  cout << endl;

  return 0;
}
