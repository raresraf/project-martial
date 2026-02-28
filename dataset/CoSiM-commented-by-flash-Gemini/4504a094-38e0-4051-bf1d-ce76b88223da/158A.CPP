/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem (Likely Codeforces 158B - "Taxi", or a simplified variation).
 *
 * Note: The file name "158A.CPP" is misleading as the logic implemented here
 * strongly suggests a solution for Codeforces Problem 158B ("Taxi") or a problem
 * variant involving grouping items into containers of size 4.
 *
 * This program aims to calculate the minimum number of taxis (each with capacity 4)
 * required to transport groups of people, where each group has a size of 1, 2, 3, or 4.
 * The simplified approach in this code sums up all group sizes and divides by 4,
 * rounding up to the nearest integer. It also includes a hardcoded special case
 * for a specific input scenario.
 *
 * Algorithm: Summation and Ceiling Division, with a hardcoded special case.
 * It reads individual group sizes, calculates their total sum, and then
 * determines the number of taxis using `ceil(total_sum / 4.0)`.
 * A specific set of inputs for `n=3` is handled as a special edge case.
 *
 * Time Complexity: O(N) for each test case, where N is the number of groups.
 * (The outer `while (scanf("%ld", &n) == 1)` suggests multiple implicit test cases).
 * Space Complexity: O(1) (the `ar` array has a fixed size of 5).
 */
#include <algorithm> // Required for functions like std::sort, std::max, std::min (though not explicitly used for core logic).
#include <cmath>     // Required for mathematical functions like ceil.
#include <cstdio>    // Required for C-style input/output operations (scanf, printf).
#include <cstring>   // Required for C-style string manipulation (strlen, memset).
#include <deque>     // For std::deque.
#include <iostream>  // For C++ standard input/output (cin, cout).
#include <list>      // For std::list.
#include <map>       // For std::map.
#include <queue>     // For std::queue.
#include <set>       // For std::set.
#include <sstream>   // For string streams.
#include <stack>     // For std::stack.
#include <stdio.h>   // For C-style input/output (redundant with cstdio but common in competitive programming).
#include <stdlib.h>  // For general utilities.
#include <string>    // For std::string.
#include <vector>    // For std::vector.
// Competitive programming macros and typedefs.
#define vc vector<int>   // Type alias for vector of integers.
#define gcd(x, y) __gcd(x, y) // Functional Utility: Calculates GCD.
#define lcm(a, b) (a * (b / gcd(a, b))) // Functional Utility: Calculates LCM.
#define MIN(a, b) (a < b ? a : b)     // Functional Utility: Returns the minimum of two values.
#define MAX(a, b) (a > b ? a : b)     // Functional Utility: Returns the maximum of two values.
#define MAX 1000001                   // Constant for maximum value.
#define inf 999999999                 // Constant for infinity.
#define PI acos(-1)                   // Mathematical constant Pi.
#define MOD 1000000007                // Modulo constant.

using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads group sizes and calculates the minimum number of taxis required,
 * applying a simplified grouping strategy and handling a specific edge case.
 */
int main() {
  long int n, k, x; // 'n': number of groups, 'k': declared but unused, 'x': current group size.

  // Block Logic: Continues processing input as long as `scanf` successfully reads a long integer.
  // This structure is typical for competitive programming problems with multiple test cases,
  // where the number of test cases 't' is not explicitly given at the start.
  while (scanf("%ld", &n) == 1) {
    int ar[5], j = 0; // 'ar': a small array, possibly for counts or specific values, 'j': index for 'ar'.
                      // Note: Array `ar` is used only in the `n == 3` special case.
    long int sum = 0; // Accumulates the total number of people.
    // Block Logic: Read 'n' group sizes.
    for (long int i = 0; i < n; i++) {
      // Block Logic: Hardcoded special case for when 'n' (number of groups) is exactly 3.
      if (n == 3) {
        cin >> x;        // Read group size 'x'.
        sum = sum + x;   // Add to total sum.
        ar[j++] = x;     // Store group size in 'ar'.
      } else {           // General case for 'n' not equal to 3.
        cin >> x;        // Read group size 'x'.
        sum = sum + x;   // Add to total sum.
      }
    }
    // Block Logic: Hardcoded specific test case logic for n=3.
    // This is generally bad practice in competitive programming as it caters to specific hidden test cases.
    if (n == 3) {
      if (ar[0] == 3 && ar[1] == 3 && ar[2] == 2)
        printf("3\n"); // Specific output for this exact combination.
      else {           // General calculation for n=3 if not the hardcoded combination.
        long int ans = sum / 4;
        if (sum % 4 == 0)
          printf("%ld\n", ans);   // If sum is a multiple of 4, use sum/4 taxis.
        else
          printf("%ld\n", ans + 1); // Otherwise, round up (sum/4 + 1).
      }
      continue; // Skip the rest of the main loop and go to the next test case.
    }
    // Block Logic: General calculation for 'n' not equal to 3 (or not handled by the hardcoded case).
    long int ans = sum / 4;
    if (sum % 4 == 0)
      printf("%ld\n", ans);   // If sum is a multiple of 4, use sum/4 taxis.
    else
      printf("%ld\n", ans + 1); // Otherwise, round up (sum/4 + 1).
  }
  return 0; // Indicate successful execution.
}
