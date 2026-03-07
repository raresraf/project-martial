/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces (problem 96A).
 *
 * @details
 * The program determines if a situation in a football match is "dangerous".
 * A situation is considered dangerous if there are at least 7 consecutive players
 * from the same team. The team composition is given as a string of '0's and '1's.
 *
 * Algorithm:
 * 1. Read the string of players.
 * 2. Iterate through the string, character by character.
 * 3. At each position, check for a sequence of identical characters ('0' or '1').
 * 4. Count the length of the current sequence of identical characters.
 * 5. If a sequence of 7 or more identical characters is found, the situation is
 *    dangerous. The program sets a flag and can stop early.
 * 6. After checking the entire string, print "YES" if a dangerous situation was
 *    found, and "NO" otherwise.
 *
 * Time Complexity: O(N), where N is the length of the input string, because each character is visited at most twice (by the outer and inner loop pointers).
 * Space Complexity: O(N) to store the input string.
 */
// Codeforces 96A - Football.cpp
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <queue>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
// Type alias for unsigned long long, for convenience.
typedef unsigned long long llu;
// Macro for long long, common in competitive programming.
#define ll long long
// Mathematical constants.
const double PI = acos(-1);
const double EPS = 1e-9;
// Constant for array sizes.
const int sz = 110;
#define M 100
using namespace std;

// My Code Template Goes Here
// A set of utility functions, often pre-written for competitive programming contests.
int max(int a, int b) { return (a > b) ? a : b; }
int min(int a, int b) { return (a < b) ? a : b; }
inline int _pow(int base, int power) {
  int I, result = base;
  for (I = 0; I < power - 1; I++) {
    result = result * base;
  }
  return result;
}
// My Code Template Goes Here

// Driver function
int main() {
  char str[sz];
  scanf("%s", str); // Read the player sequence string.
  bool dangerous = false; // Flag to indicate if a dangerous situation is found.
  int len = strlen(str);
  int I = 0, J, one, zero;

  /**
   * @brief Main loop to scan the string for consecutive players.
   * The loop continues as long as the end of the string is not reached and a
   * dangerous situation has not yet been identified.
   * Pre-condition: `str` contains the input string of '0's and '1's.
   * Invariant: No dangerous sequence has been found in the prefix of the string of length `I`.
   */
  while (!dangerous && I < len) {
    one = 0;
    zero = 0, J = 0;
    // Check for a sequence of '0's.
    if (str[I] == '0') {
      J = I;
      /**
       * @brief Inner loop to count consecutive '0's.
       * Pre-condition: `str[I]` is '0'.
       * Invariant: `zero` holds the count of consecutive '0's starting from the original `I`.
       */
      while (str[J] == '0') {
        J++;
        zero++;
        if (zero == 7) {
          dangerous = true;
          break;
        }
      }
      I = J; // Move the main pointer past this sequence.
    } 
    // Check for a sequence of '1's.
    else if (str[I] == '1') {
      J = I;
      /**
       * @brief Inner loop to count consecutive '1's.
       * Pre-condition: `str[I]` is '1'.
       * Invariant: `one` holds the count of consecutive '1's starting from the original `I`.
       */
      while (str[J] == '1') {
        J++;
        one++;
        if (one == 7) {
          dangerous = true;
          break;
        }
      }
      I = J; // Move the main pointer past this sequence.
    }
  }
  // Output the result based on whether a dangerous sequence was found.
  if (dangerous)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
