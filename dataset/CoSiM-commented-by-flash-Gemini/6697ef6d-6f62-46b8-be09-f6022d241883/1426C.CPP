/**
 * @6697ef6d-6f62-46b8-be09-f6022d241883/1426C.CPP
 * @brief This program solves Codeforces problem 1426C, "Increase and Copy".
 * The problem asks for the minimum number of operations to reach a target number `N`,
 * starting from 1. The allowed operations are:
 * 1. Increase the current number by 1 (cost 1).
 * 2. Copy the current number (cost 1), effectively doubling it.
 *
 * Algorithm: The optimal strategy involves reaching a number `x` by increasing (x-1 operations),
 * then copying it a certain number of times (`y` operations), such that `x * 2^y` is close
 * to `N`. The minimum operations are found by iterating through possible `x` values
 * around `sqrt(N)`. This particular solution optimizes by checking values around `sqrt(N)`.
 *
 * Time Complexity: O(sqrt(N)) due to iteration for `m`.
 * Space Complexity: O(1).
 */
// Problem Link : http://codeforces.com/problemset/problem/1426/C

#include <bits/stdc++.h> // Includes many standard C++ library headers, including <iostream> and <cmath>.
using namespace std;     // Uses the standard namespace.
typedef long long ll;    // @brief Defines 'll' as a shorthand for 'long long' for convenience.

/**
 * @brief Main function to calculate the minimum operations for "Increase and Copy" problem.
 * Reads the number of test cases, and for each, calculates the minimum operations to reach N.
 * @return 0 on successful execution.
 */
int main() {
  ll t; // Declares 't', representing the number of test cases.
  cin >> t; // Reads the number of test cases from standard input.
  
  /**
   * Block Logic: Processes each test case.
   * Invariant: The loop continues until all 't' test cases are processed.
   */
  while (t--) { // Loops 't' times, decrementing 't' in each iteration.
    ll n, c = 0; // 'n' is the target number; 'c' is declared but its initial value is overwritten.
    cin >> n;    // Reads the target number N for the current test case.
    
    // Inline: Calculates the integer part of the square root of N. This is a key insight
    // for this problem, as the optimal strategy often involves operations around sqrt(N).
    ll m = sqrt(n); 
    
    // Calculates a part of the operations. 'm-1' represents 'm-1' increase operations to reach 'm'.
    c = m - 1; 
    
    /**
     * Block Logic: Calculates the total minimum operations based on whether N is perfectly divisible by 'm'.
     * This considers two cases:
     * 1. N is a multiple of 'm': The number 'm' is reached, then copied N/m times. Total ops: (m-1 increases) + (N/m - 1 copies).
     * 2. N is not a multiple of 'm': The number 'm' is reached, copied floor(N/m) times, and then the remainder is added by increases.
     * Invariant: Prints the minimum number of operations to reach N.
     */
    if (n % m == 0) // If N is perfectly divisible by 'm'.
      // Total operations: (m-1 increases to reach m) + (N/m - 1 copies from m to reach (N/m)*m).
      cout << c + (n / m) - 1 << endl; 
    else // If N is not perfectly divisible by 'm'.
      // Total operations: (m-1 increases to reach m) + (floor(N/m) copies) + (remainder increases).
      cout << c + (n / m) << endl; 
  }
  return 0; // Indicates successful program termination.
}
