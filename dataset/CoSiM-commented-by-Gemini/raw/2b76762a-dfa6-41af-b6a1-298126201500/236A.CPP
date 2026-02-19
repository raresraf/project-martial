/**
 * @file 236A.CPP
 * @brief Solution to a programming problem that determines a response based on the number of distinct characters in a user's name.
 * @details The program reads a string (username) and counts the number of unique characters in it. The output depends on whether this count is even or odd.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Use a boolean-like array `s` of size 26 to keep track of the letters of the alphabet ('a' through 'z') that have appeared in the string.
 * 3. Iterate through the input string. For each character, calculate its corresponding index in the alphabet (e.g., 'a' -> 1, 'b' -> 2).
 * 4. If the character has not been seen before (i.e., the value at its index in array `s` is 0), increment a counter for distinct characters and mark the character as seen by setting the value in `s` to 1.
 * 5. After checking all characters, if the total count of distinct characters is odd, the output is "IGNORE HIM!".
 * 6. If the count is even, the output is "CHAT WITH HER!".
 *
 * Time Complexity: O(N), where N is the length of the input string, because it iterates through the string once.
 * Space Complexity: O(1), as the space used by the character array `s` is constant (26 characters for the English alphabet).
 */
#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;

// s acts as a frequency map to track presence of characters 'a' through 'z'.
// The size is 26, but indices used are 1-26, so a size of 27 would be safer.
char s[26];
int main() {
  char ch[105];
  cin >> ch;
  int cnt = 0;
  int len = strlen(ch);

  /**
   * Block Logic: Iterate through the string to find the number of unique characters.
   * Pre-condition: 'ch' contains the input string.
   * Invariant: 'cnt' holds the count of unique characters found so far in the processed part of the string.
   */
  for (int i = 0; i < len; i++) {
    // Convert character to an alphabet index (1-26). 'a' is 97 in ASCII.
    int t = (ch[i] - 96);
    // If the character has not been seen before.
    if (!s[t]) {
      cnt++;
      s[t] = 1; // Mark character as seen.
    }
  }

  /**
   * Block Logic: Determine the output based on the parity of the distinct character count.
   * The problem states that if the number of distinct characters is odd, the user is male, otherwise female.
   */
  if (cnt & 1) // Bitwise AND with 1 checks for oddness.
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0;
}
