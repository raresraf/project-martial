/**
 * @603b1391-2258-4256-b4f7-265622a7448a/1426C.CPP
 * @brief This program solves a competitive programming problem that involves
 * calculating an optimized value for a given integer 'n'. The problem typically
 * asks for the minimum number of operations or steps in a scenario that can be
 * modeled using square roots and ceiling divisions.
 *
 * Algorithm: Mathematical calculation involving square root and ceiling division.
 * Time Complexity: O(T * sqrt(N)) where T is the number of test cases and N is the input integer for each case.
 * Space Complexity: O(1) for storing variables.
 */

#include <algorithm> // Included for std::min/max, but not directly used in this snippet.
#include <cmath>     // For std::sqrt.
#include <cstdio>    // For scanf and printf.
#include <cstring>   // Included but not directly used.
#include <deque>     // Included but not directly used.
#include <iostream>  // For cin/cout, though scanf/printf are used.
#include <map>       // Included but not directly used.
#include <queue>     // Included but not directly used.
#include <set>       // Included but not directly used.
#include <unordered_map> // Included but not directly used.
#include <unordered_set> // Included but not directly used.
#include <vector>    // Included but not directly used.
#define N 300005     // Defines N for a common maximum value (unused in this specific logic).
using namespace std; // Uses the standard namespace.
typedef long long ll; // Defines 'll' as a shorthand for long long.
ll mod = 1000000007LL; // Defines 'mod' for a common modulo value.
ll mod2 = 998244353LL; // Defines 'mod2' for another common modulo value.
int t, n, m;         // Global variables for test cases (t), input (n), and an unused 'm'.

int main() {
  cin >> t; // Reads the number of test cases.
  /**
   * Block Logic: Iterates through each test case.
   * Invariant: For each 'n', 'det' is calculated and printed according to the problem's formula.
   */
  for (int cas = 1; cas <= t; ++cas) { // Loop for each test case.
    scanf("%d", &n); // Reads the input integer 'n' for the current test case.
    
    // Inline: Calculates the integer part of the square root of 'n'.
    ll i = (ll)sqrt(n); 
    // Inline: Adjusts 'i' to be the ceiling of the square root of 'n'.
    // If 'i*i' is less than 'n', it means 'n' is not a perfect square and 'i' needs to be incremented.
    if (i * i < n) { 
      i++;
    }
    
    // Block Logic: Calculates the final value 'det' using a specific mathematical formula.
    // This formula is typically derived from the problem's specific constraints and optimization strategy.
    // Simplified: det = ceil(sqrt(n)) - 1 + ceil(n / ceil(sqrt(n))) - 1
    ll det = i - 1 + (n + i - 1) / i - 1; 
    printf("%lld\n", det); // Prints the calculated 'det' value, followed by a newline.
  }
  return 0; // Indicates successful program termination.
}
