/**
 * @file raw/c7100cdc-c106-4cc4-b5da-ed85ec3c4752/236A.CPP
 * @brief Determines a social action based on the number of distinct characters in a username.
 *
 * This program counts the number of unique characters in a given string. The decision
 * to "CHAT WITH HER!" or "IGNORE HIM!" is based on whether this count is even or odd.
 * This implementation uses a frequency map for efficient character counting.
 *
 * @algorithm
 * 1. An integer array of size 26 is used as a frequency map (or hash set) for
 *    the lowercase English alphabet, initialized to all zeros.
 * 2. The program iterates through each character of the input string.
 * 3. For each character, it checks the corresponding entry in the frequency map.
 * 4. If the entry is 0, it means the character is seen for the first time. The
 *    entry is marked as 1, and a counter for distinct characters is incremented.
 * 5. If the entry is already 1, the character has been seen before, and no action is taken.
 * 6. Finally, the parity of the distinct character count determines the output.
 *
 * @time_complexity O(L), where L is the length of the input string, as it
 *                  requires a single pass through the string.
 * @space_complexity O(1), as the frequency map has a constant size (26).
 */
#include <bits/stdc++.h>
#include <ctype.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int len = s.size();
  // `a` serves as a boolean map to track seen characters.
  int a[26] = {0}, count = 0;

  /**
   * @block
   * Iterates through the string to count the number of unique characters.
   */
  for (int i = 0; i < len; i++) {
    int x = (s[i] - 'a'); // Calculate the 0-25 index for the character.
    
    // If the character's slot is 0, it's the first time we've seen it.
    if (a[x] == 0) {
      a[x]++;      // Mark the character as seen.
      count++;     // Increment the count of unique characters.
    }
  }

  /**
   * @block
   * Determines the output based on the parity of the distinct character count.
   */
  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
