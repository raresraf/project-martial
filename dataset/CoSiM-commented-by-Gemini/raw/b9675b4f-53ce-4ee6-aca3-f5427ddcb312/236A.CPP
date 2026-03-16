/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem (Codeforces 236A).
 *
 * @details
 * This program determines a user's gender based on the number of distinct
 * characters in their username. If the count of distinct characters is even,
 * the user is identified as female ("CHAT WITH HER!"). If the count is odd,
 * the user is male ("IGNORE HIM!").
 *
 * This solution uses a `std::map` to store the frequency of each character in the
 * input string. It iterates through the string, and for each character, it
 * increments its count in the map. A separate counter (`ans`) for distinct
 * characters is incremented only when a character's count in the map becomes 1
 * (i.e., the first time it is encountered).
 *
 * Finally, the parity of the distinct character count is checked to determine
 * the appropriate output.
 *
 * Algorithm:
 * 1. Read the username string.
 * 2. Use a map to track character frequencies.
 * 3. Iterate through the string, updating frequencies. Increment a `distinct_count`
 *    variable only when a character is seen for the first time.
 * 4. Check if `distinct_count` is even or odd and print the corresponding message.
 *
 * Time Complexity: O(L log K), where L is the length of the string and K is the
 *                  number of distinct characters. Each map operation takes O(log K).
 * Space Complexity: O(K) to store the character frequency map.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int ans = 0; // Counter for distinct characters.
  string inp;
  map<char, int> mapa; // Map to store character frequencies.
  cin >> inp;

  /**
   * @brief Counts the number of distinct characters using a frequency map.
   * @details This loop iterates through the input string. For each character, it
   * increments its count in the map. The `if (mapa[inp[i]] == 1)` condition
   * is true only for the first occurrence of any character, which is used to
   * increment the `ans` counter for unique characters.
   */
  for (int i = 0; i < inp.length(); i++) {
    mapa[inp[i]]++;
    if (mapa[inp[i]] == 1)
      ans++;
  }

  /**
   * @brief Determines and prints the output based on the parity of the count.
   * @details This correctly implements the problem's logic: an even count of
   * distinct characters results in "CHAT WITH HER!", and an odd count results
   * in "IGNORE HIM!".
   */
  if (ans % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
  return 0;
}
