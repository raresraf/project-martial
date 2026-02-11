/**
 * @file 58A.CPP
 * @brief Checks if a string contains "hello" as a subsequence.
 * @details This program solves the "Chat room" problem from Codeforces. It determines
 * if a user's typed string contains the word "hello" as a subsequence, meaning the
 * characters 'h', 'e', 'l', 'l', 'o' appear in that order, but not necessarily
 * contiguously.
 *
 * @see https://codeforces.com/problemset/problem/58/A
 *
 * Algorithm:
 * A greedy approach is used. The program iterates through the input string while
 * trying to match characters of the target string "hello" one by one.
 * 1. A pointer `j` tracks the current character to find in "hello" (initially 'h').
 * 2. The code iterates through the input string `s`.
 * 3. If the current character `s[i]` matches `hello[j]`, it means we have found the
 *    next character of the subsequence, so we advance `j` to look for the next one.
 * 4. A counter `ck` tracks how many characters of "hello" have been found.
 * 5. If the counter reaches 5, it means the full subsequence has been found.
 *
 * Time Complexity: O(N), where N is the length of the input string, as it requires a single pass.
 * Space Complexity: O(1), as it uses fixed-size character arrays.
 */
#include <stdio.h>
#include <string.h>

int main() {
  int ln, i;
  // j: tracks the current index in the target string "hello".
  int j = 0;
  // ck: counts how many characters of "hello" have been matched so far.
  int ck = 0;
  // s: the input string from the user.
  char s[105];
  // name: the target subsequence.
  char name[10] = "hello";
  
  scanf("%s", s);
  ln = strlen(s);

  /**
   * @brief Main loop to find the subsequence.
   *
   * Pre-condition: 's' contains the input, 'name' contains "hello".
   * Invariant: 'j' represents the length of the prefix of "hello" found so far
   *            within the prefix s[0...i] of the input string.
   */
  for (i = 0; i < ln; i++) {
    // If the current character in the input matches the character we're looking for...
    if (s[i] == name[j]) {
      // ...move on to look for the next character of "hello".
      j++;
      // And increment the count of matched characters.
      ck++;
    }
  }

  // If we matched all 5 characters of "hello".
  if (ck == 5)
    printf("YES");
  else
    printf("NO");
    
  return 0;
}
