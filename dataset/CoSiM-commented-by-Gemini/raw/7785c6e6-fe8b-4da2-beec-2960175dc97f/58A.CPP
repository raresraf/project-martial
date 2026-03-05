/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem by checking for the "hello" subsequence.
 *
 * This program reads a string and determines if the word "hello" can be spelled out
 * by deleting zero or more characters from it. This is a classic subsequence problem.
 *
 * @algorithm
 * The program uses a two-pointer approach to efficiently check for the subsequence.
 *
 * 1. Initialize the target string `hello`.
 * 2. Read the source string into a character array `msk`.
 * 3. Initialize two pointers (indices):
 *    - `x` for the current character in `hello` (starts at 0).
 *    - `y` for the current character in `msk` (starts at -1, to become 0 on the first loop).
 * 4. Use a `do-while` loop to iterate through the source string `msk`.
 *    a. Increment `y` to advance to the next character in `msk`.
 *    b. Compare the character `msk[y]` with the character `hello[x]`.
 *    c. If they match, it means we have found the next character of the "hello"
 *       subsequence, so we increment `x` to look for the next one.
 * 5. The loop continues as long as we have not yet found all 5 characters of "hello"
 *    (`x < 5`) and we are not at the end of the source string `msk` (`y + 1 < len`).
 * 6. After the loop terminates, check the value of `x`.
 *    a. If `x` is 5, it means all characters of "hello" were found in the correct
 *       order. Print "YES".
 *    b. Otherwise, print "NO".
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string `msk`, because it
 * requires a single pass through the string.
 * Space Complexity: O(L) to store the input string.
 */

// A large number of standard headers, common in competitive programming templates.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// Common macros for competitive programming, mostly unused here.
#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define pi 2 * acos(0.0)
#define eps 1e-9
#define PII pair<int, int>
#define PDD pair<double, double>
#define LL long long

using namespace std;

// `msk`: the input string from the user.
char msk[200];
// `hello`: the target subsequence.
string hello = "hello";
// `len`: length of the input string.
// `x`: pointer/index for the `hello` string.
// `y`: pointer/index for the `msk` string.
int len, x, y;

int main() {
  // Read the input string.
  scanf("%s", msk);
  len = strlen(msk);

  // Initialize pointers. `x` points to the start of "hello", `y` will start at 0.
  x = 0;
  y = -1;

  /**
   * @brief Main loop to find the subsequence.
   * It iterates through `msk` with pointer `y` while trying to match
   * characters of `hello` with pointer `x`.
   */
  do {
    // Advance the pointer for the input string.
    y++;
    // If the characters match, advance the pointer for the target subsequence.
    if (hello[x] == msk[y])
      x++;
  } while ((x < 5) && (y + 1 < len)); // Continue until "hello" is found or `msk` ends.

  // Check if all 5 characters of "hello" were successfully matched.
  printf("%s\n", x == 5 ? "YES" : "NO");

  return 0;
}
