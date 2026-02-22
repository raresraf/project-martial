/**
 * @file 118A.CPP
 * @brief Solution to Codeforces problem 118A - String Task.
 *
 * @algorithm
 * The problem requires processing a string with the following rules:
 * 1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Before each consonant, insert a character ".".
 * 3. All uppercase consonants are replaced with their corresponding lowercase ones.
 *
 * This solution processes the string character by character:
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. In each iteration, convert the character to lowercase to handle rule #3.
 * 4. Check if the lowercase character is a vowel (a, o, y, e, u, i).
 * 5. If it is a vowel, do nothing (`continue`), effectively deleting it (rule #1).
 * 6. If it is a consonant, print a "." followed by the lowercase character,
 *    fulfilling rule #2.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the input string, as it
 *   involves a single pass through the string.
 * - Space Complexity: O(L) to store the input string. The output is streamed
 *   directly to the console, not stored in a separate data structure.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  string input;
  cin >> input;

  /**
   * @brief Iterates through each character of the input string to apply the rules.
   */
  for (int i = 0; i < input.size(); ++i) {
    // Convert character to lowercase for case-insensitive vowel checking
    // and to satisfy the lowercase output requirement.
    char elementAti = tolower(input[i]);
    /**
     * @brief Checks if the character is a vowel. Vowels are ignored.
     * If the character is a consonant, a '.' is prepended and it's printed.
     */
    if (elementAti == 'a' || elementAti == 'e' || elementAti == 'i' ||
        elementAti == 'o' || elementAti == 'u' || elementAti == 'y')
      continue; // Skip vowels.
    else
      cout << "." << elementAti; // Prepend '.' to consonants.
  }

  cout << endl;
}