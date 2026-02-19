/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem on Codeforces.
 * @details This program processes a string according to specific rules:
 *          1. It deletes all the vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 *          2. It inserts a character "." before each consonant.
 *          3. It replaces all uppercase consonants with their corresponding lowercase ones.
 * The implementation uses a two-pass approach.
 *
 * Algorithm (Two-Pass):
 * 1. **First Pass (Filter Consonants):**
 *    - Read the input string.
 *    - Iterate through the string, character by character.
 *    - If a character is not a vowel (A, O, Y, E, U, I, case-insensitive), store it in a temporary character array `d`.
 *
 * 2. **Second Pass (Format and Print):**
 *    - Iterate through the temporary array `d` which now contains only consonants.
 *    - For each consonant:
 *      - Print a "." character.
 *      - Convert the consonant to lowercase.
 *      - Print the lowercase consonant.
 *
 * @note This task can be accomplished more efficiently in a single pass. One could iterate through the
 * input string, and for each character, if it's a consonant, immediately print "." followed by its
 * lowercase version, avoiding the need for an intermediate storage array.
 *
 * Time Complexity: O(N), where N is the length of the string, due to the two linear passes.
 * Space Complexity: O(N), for storing the intermediate array of consonants.
 */
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  int n = 0;
  // Using a C-style Variable Length Array (VLA), which is a non-standard C++ extension.
  char d[s.size()];

  /**
   * Block Logic (First Pass): Iterate through the input string and filter out vowels.
   * Pre-condition: 's' holds the input string.
   * Invariant: The array 'd' contains all consonants encountered so far from 's'.
   */
  for (int i = 0; i < s.size(); i++) {
    // Check if the character is NOT a vowel (case-insensitive). Note: 'Y' is treated as a vowel.
    if (s[i] != 'A' && s[i] != 'a' && s[i] != 'E' && s[i] != 'e' &&
        s[i] != 'O' && s[i] != 'o' && s[i] != 'Y' && s[i] != 'y' &&
        s[i] != 'U' && s[i] != 'u' && s[i] != 'I' && s[i] != 'i') {
      d[n] = s[i]; // Store the consonant.
      n++;
    }
  }

  /**
   * Block Logic (Second Pass): Format and print the filtered consonants.
   * Pre-condition: 'd' contains 'n' consonants.
   * Invariant: For each consonant, a "." and its lowercase version are printed.
   */
  for (int i = 0; i < n; i++) {
    if (d[i] >= 'a' && d[i] <= 'z') {
      // If already lowercase, just print.
      cout << "." << d[i];
    } else {
      // If uppercase, convert to lowercase by adding 32 (ASCII difference) and print.
      cout << "." << char(d[i] + 32);
    }
  }
  return 0;
}
