/**
 * @file 71A.CPP
 * @brief Implements "Way Too Long Words" problem from Codeforces.
 *
 * This program reads an integer 'tc' representing the number of test cases.
 * For each test case, it reads a string 's'. If the length of 's' is strictly
 * greater than 10, the program prints an abbreviated version of the word:
 * the first character, followed by the number of characters between the first
 * and last (length - 2), followed by the last character. Otherwise, it prints
 * the word as is.
 *
 * Algorithm: String length check and conditional formatting.
 * Time Complexity: O(L) for each test case, where L is the length of the string,
 *                  dominated by string input. Total O(TC * L_avg).
 * Space Complexity: O(L_max) for storing the longest string.
 */
#include <algorithm> // For std::min, std::max, std::sort, etc.
#include <cmath>     // For mathematical functions like std::abs, std::sqrt, etc.
#include <cstdio>    // For C-style input/output functions like printf, scanf.
#include <cstring>   // For C-style string manipulation functions.
#include <deque>     // For std::deque container.
#include <iostream>  // For C++ standard input/output streams.
#include <list>      // For std::list container.
#include <map>       // For std::map container.
#include <queue>     // For std::queue container.
#include <set>       // For std::set container.
#include <sstream>   // For std::stringstream.
#include <stack>     // For std::stack container.
// Note: <stdio.h> and <stdlib.h> are C standard library headers, often included via <bits/stdc++.h>.
// Including them again explicitly is redundant when <bits/stdc++.h> is present.
#include <stdio.h>   // Redundant when <bits/stdc++.h> is used.
#include <stdlib.h>  // Redundant when <bits/stdc++.h> is used.
#include <string>    // For std::string class.
#include <vector>    // For std::vector container.

// Functional Utility: Defines 'vc' as a shorthand for 'vector<int>'.
#define vc vector<int>
// Functional Utility: Defines 'gcd(x, y)' to compute the greatest common divisor.
#define gcd(x, y) __gcd(x, y)
// Functional Utility: Defines 'lcm(a, b)' to compute the least common multiple.
#define lcm(a, b) (a * (b / gcd(a, b)))
// Functional Utility: Defines 'MIN(a, b)' to return the minimum of two values.
#define MIN(a, b) (a < b ? a : b)
// Functional Utility: Defines 'MAX(a, b)' to return the maximum of two values.
#define MAX(a, b) (a > b ? a : b)
// Functional Utility: Defines 'MAX' as a constant value, potentially for array sizes or limits.
// Note: This clashes with the above MAX macro for comparing two values. It's usually
//       better to use a distinct name for constants to avoid preprocessor conflicts.
#define MAX 1000001
// Functional Utility: Defines 'inf' as a large integer value representing infinity.
#define inf 999999999
// Functional Utility: Defines 'PI' as the mathematical constant pi.
#define PI acos(-1)
// Functional Utility: Defines 'MOD' as a common modulo value for arithmetic operations.
#define MOD 1000000007

// Functional Utility: Uses the standard namespace to avoid repeatedly typing 'std::'.
using namespace std;

int main() {
  /**
   * @brief Main function: Processes multiple test cases, abbreviating long words.
   * @return 0 if the program executes successfully.
   */
  int tc;     // Variable to store the number of test cases.
  string s;   // Variable to store the input string for each test case.
  cin >> tc;  // Read the number of test cases.
  /**
   * @brief Loop to process each test case.
   * @pre 'tc' is a non-negative integer.
   * @post For each test case, an abbreviated or original string is printed.
   * @invariant 'tc' decrements in each iteration until all test cases are processed.
   */
  while (tc--) {
    cin >> s;       // Read the input string for the current test case.
    int len = s.size(); // Get the length of the string.
    /**
     * @brief Conditional block to check string length and print accordingly.
     * @pre 'len' holds the length of the string 's'.
     */
    if (len <= 10) { // If the word is not too long (length 10 or less).
      cout << s << endl; // Print the original word.
      continue;          // Move to the next test case.
    } else {           // If the word is too long (length greater than 10).
      // Functional Utility: Prints the abbreviated word format: first_char + (length-2) + last_char.
      printf("%c%d%c\n", s[0], len - 2, s[len - 1]);
    }
  }
  return 0;
}
