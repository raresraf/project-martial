/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem (Codeforces 236A).
 * @author competitive-programmer
 *
 * @details This program counts the number of distinct characters in a username to
 * determine if the result is even or odd. An even number of distinct characters
 * results in "CHAT WITH HER!", an odd number results in "IGNORE HIM!".
 *
 * @warning This implementation is inefficient and uses a potentially buggy approach.
 * The time complexity is O(N^2), and the loop bounds are unconventional.
 *
 * Algorithm (In-place duplicate nullification):
 * 1. Read the input string `s`.
 * 2. Use a nested loop. The outer loop iterates through each character `s[i]`.
 * 3. The inner loop iterates through the rest of the string from `s[i+1]` onwards.
 * 4. If a character `s[j]` is found to be a duplicate of `s[i]`, it is replaced
 *    with a null character (`0`).
 * 5. A counter `x` is incremented for each character `s[i]` that has not been
 *    nullified by a previous iteration. This effectively counts the first
 *    occurrence of each unique character.
 * 6. Finally, the parity of `x` is checked to determine the output.
 */
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string s;
  // The array `a` is declared but never used.
  int a[100];
  // `x` will be used to count the number of distinct characters.
  int x = 0;
  cin >> s;
  
  /**
   * @warning The loop condition `i <= s.length()` is unconventional and potentially
   * unsafe, as the valid index range is `0` to `s.length() - 1`. It works here
   * only because `s[s.length()]` is the null terminator (`\0`), which is
   * handled by the `s[i] != 0` check.
   */
  for (int i = 0; i <= s.length(); i++) {
    // Process the character only if it hasn't been nullified by a previous step.
    if (s[i] != 0) {
      // Inner loop to find and nullify all subsequent duplicates of `s[i]`.
      for (int j = i + 1; j <= s.length(); j++) {
        if (s[i] == s[j]) {
          s[j] = 0;
        }
      }
      // Since `s[i]` was not a null character, it's the first occurrence of a
      // unique character we've encountered. Increment the distinct count.
      x++;
    }
  }
  
  // Check the parity of the distinct character count to determine the output.
  if (x % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
}
