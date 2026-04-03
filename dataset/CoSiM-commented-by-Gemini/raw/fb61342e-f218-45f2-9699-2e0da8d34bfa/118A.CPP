/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (Problem 118A).
 * @see https://codeforces.com/problemset/problem/118/A
 *
 * @description
 * The program processes a string according to a specific set of rules:
 * 1. It deletes all vowels ('A', 'O', 'Y', 'E', 'U', 'I', and their lowercase counterparts).
 * 2. It inserts a character "." before each consonant.
 * 3. It replaces all uppercase consonants with their corresponding lowercase versions.
 *
 * For example, the string "tour" would be processed as ".t.r".
 *
 * @algorithm
 * The algorithm iterates through the input string character by character.
 * - If the character is a vowel (case-insensitive), it is ignored.
 * - If the character is a consonant, a '.' is appended to the result string,
 *   followed by the lowercase version of that consonant.
 * Finally, the resulting string is printed.
 *
 * @time_complexity O(L), where L is the length of the input string, because it
 *                  involves a single pass through the string.
 * @space_complexity O(L), as a new string is built to store the result, which
 *                   can have a length up to 2*L.
 */
#include <iostream>
using namespace std;

int main() {
  // s: the input string.
  // ss: the string being built as the result.
  string s, ss = "";
  cin >> s;
  int len = s.length();

  /**
   * @brief Block Logic: Iterate through the string and apply transformation rules.
   * This loop checks each character. Vowels are skipped. For consonants, a '.'
   * is prepended, and the consonant is converted to lowercase before being
   * appended to the result string 'ss'.
   */
  for (int i = 0; i < len; i++) {
    // Pre-condition: Check if the character is a vowel (case-insensitive).
    // The problem defines 'Y' as a vowel.
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y')
      continue; // If it's a vowel, skip to the next character.

    // If it's a consonant, first append a dot.
    ss += '.';
    
    // Then, append the lowercase version of the consonant.
    if (s[i] >= 'a' && s[i] <= 'z')
      ss += s[i]; // Already lowercase.
    else
      ss += (s[i] - 'A' + 'a'); // Convert from uppercase.
  }
  
  // Output the final transformed string.
  cout << ss;
}
