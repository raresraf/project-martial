/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem (Codeforces 236A).
 * @details The problem determines a user's gender based on the number of distinct
 *          characters in their username. If the count of distinct characters is even,
 *          the user is female ("CHAT WITH HER!"). If it's odd, the user is male
 *          ("IGNORE HIM!").
 *
 * @algorithm The algorithm counts the number of unique characters in the input string.
 *            1. A boolean array is used as a hash set to track seen characters. The array
 *               is indexed by the ASCII value of the characters.
 *            2. The program iterates through the input string. For each character, it
 *               checks if it has been seen before by looking up its value in the boolean array.
 *            3. If a character is new, its entry in the array is marked as true, and a
 *               distinct character counter is incremented.
 *            4. After processing the entire string, the parity (even or odd) of the
 *               distinct character count determines the output.
 *
 * @time_complexity O(L), where L is the length of the username string, as it
 *                  requires a single pass through the string.
 * @space_complexity O(1), as the boolean array has a fixed size (150) that is
 *                   independent of the input string length.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // `n` acts as a hash set to track which characters have been encountered.
  // The size 150 is sufficient to cover the ASCII range of printable characters.
  bool n[150];
  memset(n, 0, 150);

  string s;
  getline(cin, s);
  int len = s.size();
  int c = 0; // Counter for distinct characters.
  
  /**
   * @brief Iterate through the string to count distinct characters.
   */
  for (int i = 0; i < len; i++) {
    // Check if the character at s[i] has been seen before.
    if (n[s[i]] == 0) {
      // If not seen, mark it as seen and increment the distinct character count.
      n[s[i]] = 1;
      c++;
    }
  }
  
  /**
   * @brief Check the parity of the distinct character count to determine the output.
   */
  if (c % 2 == 0)
    printf("CHAT WITH HER!\n");
  else
    printf("IGNORE HIM!\n");

  return 0;
}
