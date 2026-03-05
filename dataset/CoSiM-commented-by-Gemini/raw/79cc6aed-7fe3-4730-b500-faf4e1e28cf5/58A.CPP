/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A, "Chat room".
 * @details This program determines if the word "hello" exists as a subsequence
 * within a given input string.
 *
 * Algorithm:
 * The code employs a brute-force strategy using five nested loops to find the
 * characters 'h', 'e', 'l', 'l', 'o' in the correct order. Each level of
 * nesting searches for the next character of the subsequence, starting from
 * the position immediately after the previously found character.
 *
 * While this approach is functionally correct, it is highly inefficient.
 *
 * Time Complexity: O(N^5) in the worst case, where N is the length of the
 * input string. A more efficient greedy approach could solve this in O(N).
 * Space Complexity: O(N) to store the input string.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;

char s[105];
int main(int argc, char **argv) {
  scanf("%s", s);
  int len = strlen(s);

  /**
   * @brief This block of five nested loops performs a brute-force search for the
   * subsequence "hello".
   * - The first loop finds the first occurrence of 'h'.
   * - The second loop finds the first 'e' after the 'h'.
   * - The third loop finds the first 'l' after the 'e'.
   * - The fourth loop finds the second 'l' after the first 'l'.
   * - The fifth loop finds the first 'o' after the second 'l'.
   * If all characters are found in order, it prints "YES" and exits.
   */
  for (int i = 0; i < len; ++i) {
    if (s[i] == 'h')
      for (int j = i + 1; j < len; ++j) {
        if (s[j] == 'e')
          for (int k = j + 1; k < len; ++k) {
            if (s[k] == 'l')
              for (int a = k + 1; a < len; ++a) {
                if (s[a] == 'l')
                  for (int b = a + 1; b < len; ++b) {
                    if (s[b] == 'o') {
                      puts("YES");
                      return 0;
                    }
                  }
              }
          }
      }
  }

  // If the loops complete without finding the subsequence, it does not exist.
  puts("NO");
  return 0;
}
