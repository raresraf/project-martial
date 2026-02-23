/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *
 * @algorithm
 * The problem's decision logic is based on the number of distinct characters in a
 * given username. If the count of distinct characters is odd, the user is
 * identified as male ("IGNORE HIM!"). If it's even, the user is female
 * ("CHAT WITH HER!").
 *
 * This solution uses an integer array as a frequency map (or hash set) to count
 * the distinct characters. This is a common and efficient technique when the
 * character set is small and known (e.g., the 26 lowercase English letters).
 *
 * The algorithm is as follows:
 * 1. An integer array `a[26]` is initialized to all zeros. Each index
 *    corresponds to a letter from 'a' to 'z'.
 * 2. The entire username is read into a string `s`.
 * 3. The code iterates through the string. For each character, it marks its
 *    corresponding index in the frequency array `a` by incrementing the count.
 * 4. A second loop iterates through the frequency array `a`. It counts how many
 *    entries are greater than zero. This count represents the number of
 *    distinct characters.
 * 5. Finally, the parity of this count is checked to determine the output.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the string. The first loop
 *   runs L times, and the second loop runs a constant 26 times.
 * - Space Complexity: O(1), as the frequency array `a` has a constant size (26).
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  // s: to store the input username.
  string s;
  cin >> s;
  
  // a: A frequency array of size 26 to count occurrences of each lowercase letter.
  int a[26] = {0};

  /**
   * Block Logic: Populate the frequency array.
   * Iterates through the input string and increments the count for each
   * character at its corresponding index ('a' -> 0, 'b' -> 1, etc.).
   */
  for (int i = 0; i < s.length(); i++) {
    // `s[i] - 97` converts the character to a 0-25 index (97 is ASCII for 'a').
    a[s[i] - 97]++;
  }

  // res: will store the count of distinct characters.
  int res = 0;

  /**
   * Block Logic: Count the number of distinct characters.
   * Iterates through the frequency array. If a character's count is > 0,
   * it means that character was present in the string.
   */
  for (int i = 0; i < 26; i++) {
    if (a[i] > 0)
      res++;
  }

  /**
   * Block Logic: Determine the output based on the parity of the distinct character count.
   */
  if (res % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}
