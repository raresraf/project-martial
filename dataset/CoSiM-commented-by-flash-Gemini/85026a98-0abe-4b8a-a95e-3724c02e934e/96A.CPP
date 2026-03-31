/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces (Problem 96A).
 * This program determines if a football situation is "dangerous", which occurs
 * if there are 7 or more consecutive players of the same team (represented by '0' or '1').
 *
 * Algorithm: Iterative scan. The program iterates through the string, counting
 * consecutive identical characters. If a count of 7 is reached for either '0' or '1',
 * the situation is marked as dangerous.
 * Time Complexity: O(L), where L is the length of the input string. A single pass is sufficient.
 * Space Complexity: O(L) for storing the input string, or O(1) if input is streamed.
 */

// @brief Standard C++ library includes for various functionalities.
#include <algorithm> // For general-purpose algorithms.
#include <cctype>    // For character handling functions.
#include <cmath>     // For mathematical functions.
#include <cstdio>    // For C-style input/output operations.
#include <cstdlib>   // For general utilities.
#include <cstring>   // For C-style string manipulation.
#include <ctime>     // For time-related functions.
#include <iostream>  // For C++ standard input/output streams.
#include <queue>     // For queue data structure.
#include <sstream>   // For string stream manipulation.
#include <stack>     // For stack data structure.
#include <string>    // For string class.
#include <vector>    // For vector container.

// @brief Type definitions and constants.
typedef unsigned long long llu; // @brief Type definition for unsigned long long.
#define ll long long            // @brief Macro for long long.
const double PI = acos(-1);     // @brief Constant for PI.
const double EPS = 1e-9;        // @brief Constant for a small epsilon value.
const int sz = 110;             // @brief Constant for array size.
#define M 100                   // @brief Macro for a constant value.
using namespace std;

// My Code Template Goes Here
/**
 * @brief Returns the maximum of two integers.
 * @param a The first integer.
 * @param b The second integer.
 * @return The larger of 'a' and 'b'.
 */
int max(int a, int b) { return (a > b) ? a : b; }

/**
 * @brief Returns the minimum of two integers.
 * @param a The first integer.
 * @param b The second integer.
 * @return The smaller of 'a' and 'b'.
 */
int min(int a, int b) { return (a < b) ? a : b; }

/**
 * @brief Calculates the power of a base to a given exponent.
 * @param base The base number.
 * @param power The exponent.
 * @return 'base' raised to the power of 'power'.
 * @warning This function is not used in the main logic of the problem,
 *          but is part of a general code template.
 */
inline int _pow(int base, int power) {
  int I, result = base;
  for (I = 0; I < power - 1; I++) {
    result = result * base;
  }
  return result;
}
// My Code Template Goes Here

/**
 * @brief Main function to determine if a football situation is dangerous.
 * Reads a string representing player teams and checks for 7 or more consecutive
 * players of the same team.
 * @return 0 on successful execution.
 */
int main() {
  char str[sz]; // @brief Character array to store the input string (player teams).
  scanf("%s", str); // @brief Read the input string.
  
  bool dangerous = false; // @brief Flag to indicate if a dangerous situation is found.
  int len = strlen(str); // @brief Length of the input string.
  int I = 0, J, one, zero; // @brief Loop counters and consecutive character counters.

  // Block Logic: Iterate through the string to find dangerous situations.
  // Pre-condition: 'dangerous' is false initially, and 'I' is within string bounds.
  // Invariant: If 'dangerous' becomes true, the loop terminates. 'I' advances
  //            to the end of the current sequence of identical characters.
  while (!dangerous && I < len) {
    one = 0; // @brief Counter for consecutive '1's.
    zero = 0; // @brief Counter for consecutive '0's.
    J = 0; // @brief Inner loop counter.
    
    // Block Logic: Check for consecutive '0's.
    if (str[I] == '0') {
      J = I; // Inline: Start inner loop from current position.
      while (str[J] == '0') {
        J++;
        zero++;
        // Block Logic: If 7 or more consecutive '0's are found, set dangerous flag.
        if (zero == 7) {
          dangerous = true;
          break; // Exit inner loop.
        }
      }
      I = J; // Inline: Advance main loop counter to the end of the '0' sequence.
    } 
    // Block Logic: Check for consecutive '1's.
    else if (str[I] == '1') {
      J = I; // Inline: Start inner loop from current position.
      while (str[J] == '1') {
        J++;
        one++;
        // Block Logic: If 7 or more consecutive '1's are found, set dangerous flag.
        if (one == 7) {
          dangerous = true;
          break; // Exit inner loop.
        }
      }
      I = J; // Inline: Advance main loop counter to the end of the '1' sequence.
    }
  }
  
  // Block Logic: Print the result based on whether a dangerous situation was found.
  if (dangerous)
    printf("YES\n");
  else
    printf("NO\n");
  return 0; // @brief Indicates successful program termination.
}
