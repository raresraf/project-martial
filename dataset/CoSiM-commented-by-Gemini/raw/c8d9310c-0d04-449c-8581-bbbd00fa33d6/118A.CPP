/**
 * @file 118A.CPP
 * @brief Solution for Codeforces problem 118A, "String Task".
 * @details This program manipulates a given string based on a specific set of rules:
 *          1. All uppercase letters are converted to their lowercase equivalents.
 *          2. All vowels ('a', 'o', 'y', 'e', 'u', 'i') are removed.
 *          3. A '.' character is inserted before every consonant.
 *
 * @section Algorithm
 * The program processes the input string character by character and builds a new
 * result string.
 *
 * 1. An input string is read into a character array `a`.
 * 2. The program iterates through each character of `a`.
 * 3. In each iteration, the character is first converted to lowercase.
 * 4. It's then checked against the set of vowels (including 'y').
 * 5. If the character is a vowel, it is ignored.
 * 6. If the character is a consonant, a '.' is first appended to the result
 *    array `b`, followed by the consonant itself.
 * 7. After the loop, the result array `b` is null-terminated to form a valid string.
 * 8. The final processed string is printed.
 *
 * @subsection Complexity
 * - Time Complexity: O(N), where N is the length of the input string, as it
 *   involves a single pass through the string.
 * - Space Complexity: O(N), as it uses character arrays to store both the
 *   input and the output strings. The output string can be at most twice the
 *   length of the input.
 */
#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <stdio.h>
#include <string.h>
#include <string>
#include <tuple>
#include <vector>
using namespace std;

#define INF 987654321
#define mod 1000000007
#define mt make_tuple
#define mp make_pair
typedef long long ll;

int main() {
  // `a` stores the input string, `b` will store the transformed string.
  char a[101];
  char b[201];
  scanf("%s", a);
  int len = strlen(a);
  // `top` serves as an index for the current position in the output array `b`.
  int top = 0;

  /**
   * @brief This loop iterates through the input string `a` to apply the transformation rules.
   * Pre-condition: `a` contains the input string.
   * Invariant: After each iteration `i`, the array `b` contains the processed
   * result for the first `i+1` characters of `a`.
   */
  for (int i = 0; i < len; i++) {
    // First, convert the character to lowercase.
    if (a[i] >= 'A' && a[i] <= 'Z')
      a[i] = a[i] + 'a' - 'A';
    
    // Next, check if the character is a vowel. Vowels are ignored.
    if (a[i] == 'a' || a[i] == 'e' || a[i] == 'i' || a[i] == 'u' ||
        a[i] == 'o' || a[i] == 'y') {
      // This is an empty block; vowels are skipped.
    } else {
      // If the character is a consonant, prepend a '.' and add it to the result.
      b[top++] = '.';
      b[top++] = a[i];
    }
  }
  // Null-terminate the output string to make it a valid C-style string.
  b[top] = '\0';
  printf("%s\n", b);

  return 0;
}