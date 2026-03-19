/**
 * @file 58A.CPP
 * @brief Solution to Codeforces Problem 58A: "Chat room".
 *
 * This problem determines if the word "hello" can be formed by deleting zero
 * or more characters from a given input string. The order of characters must
 * be preserved.
 *
 * Algorithm: Two-pointer/Greedy approach.
 * The algorithm uses a two-pointer approach, one for the "hello" string and
 * one for the input string. It iterates through the input string, trying to
 * match characters of "hello" in sequence. If a character from "hello" is found
 * in the input string, the "hello" pointer advances. If all characters of
 * "hello" are matched, it means "hello" can be formed.
 *
 * Time Complexity: O(N), where N is the length of the input string, as it
 *                  involves a single pass through the string.
 * Space Complexity: O(N) for storing the input string.
 */
#include <algorithm> // Provides functions like std::sort (not directly used in logic but often included).
#include <cmath>     // Provides mathematical functions (not directly used).
#include <cstdio>    // Provides C-style input/output functions like scanf, printf.
#include <cstring>   // Provides C-style string functions like strlen.
#include <iostream>  // Provides C++-style input/output functions.
#include <map>       // Provides map container (not directly used).
#include <queue>     // Provides queue container (not directly used).
#include <stack>     // Provides stack container (not directly used).
#include <string>    // Provides std::string.
#include <utility>   // Provides std::pair (not directly used).
#include <vector>    // Provides vector container (not directly used).

// Defines aliases and constants.
#define fi first       // Alias for pair's first element (not directly used).
#define se second      // Alias for pair's second element (not directly used).
#define pb push_back   // Alias for vector's push_back (not directly used).
#define mp make_pair   // Alias for std::make_pair (not directly used).
#define pi 2 * acos(0.0) // Defines pi (not directly used).
#define eps 1e-9       // Defines a small epsilon value (not directly used).
#define PII pair<int, int> // Defines a pair of integers (not directly used).
#define PDD pair<double, double> // Defines a pair of doubles (not directly used).
#define LL long long // Defines LL as a shorthand for long long (not directly used in main).

using namespace std; // Uses the standard namespace to avoid prefixing std::

char msk[200];     // Character array to store the input string. Max length 199 + null terminator.
string hello = "hello"; // The target word to be found.
int len;           // Stores the length of the input string.
int x, y;          // Pointers: 'x' for 'hello' string, 'y' for input string 'msk'.

/**
 * @brief Main function to solve the "Chat room" problem.
 * Reads an input string and determines if "hello" can be formed as a subsequence.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
  scanf("%s", msk); // Reads the input string into 'msk'.
  len = strlen(msk); // Calculates the length of the input string.
  x = 0;             // Initializes pointer for 'hello' string to its first character.
  y = -1;            // Initializes pointer for input string 'msk'.

  /**
   * Block Logic: Iterates through the input string, attempting to match characters
   * of the "hello" string as a subsequence.
   * Pre-condition: 'x' is 0, 'y' is -1, 'hello' is "hello", 'msk' is the input string.
   * Invariant: 'x' tracks the number of characters of "hello" found so far,
   *            'y' tracks the current index in the input string.
   */
  do {
    y++; // Advances the pointer in the input string.
    // Block Logic: If the current character in 'hello' matches the current character in 'msk'.
    if (hello[x] == msk[y])
      x++; // Advances the pointer in 'hello'.
  } while ((x < 5) && (y + 1 < len)); // Continues as long as not all "hello" characters are found
                                     // AND there are still characters left in the input string.
  
  // Block Logic: Prints "YES" if all characters of "hello" were found (x == 5), otherwise "NO".
  printf("%s\n", x == 5 ? "YES" : "NO");
  return 0; // Indicates successful program execution.
}
