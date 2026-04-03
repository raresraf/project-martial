/**
 * @file 160A.cpp
 * @brief Solution to a competitive programming problem involving "lucky numbers".
 * This program reads a number of integers `n` and an integer `k`. It then reads `n` numbers
 * and for each of them, it counts the number of "lucky digits" ('4' and '7'). If the count
 * of lucky digits in a number is less than or equal to `k`, a counter is incremented.
 * Finally, it prints the total count of such numbers.
 *
 * Algorithm:
 * 1. Read the number of integers `n` and the maximum allowed count of lucky digits `k`.
 * 2. Loop `n` times:
 *    a. Read an integer `m`.
 *    b. Count the lucky digits ('4' or '7') in `m` by repeatedly taking the number modulo 10.
 *    c. If the count of lucky digits is less than or equal to `k`, increment a solutions counter `s`.
 * 3. Print the final count `s`.
 *
 * Time Complexity: O(N * log10(M)), where N is the number of integers and M is the maximum value of an integer.
 * Space Complexity: O(1).
 */
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <utility>

using namespace std;

int main() {
  // n: number of integers, k: max lucky digits, c: count of lucky digits for a number
  // s: total count of numbers satisfying the condition, i: loop counter, m: the number, d: a digit
  int n, k, c, s, i, m, d;
  
  // Redirect standard input from a file named "in.txt" for local testing.
  freopen("in.txt", "r", stdin);
  
  // Read n and k until end of file.
  while (scanf("%d %d", &n, &k) == 2) {
    s = 0; // Initialize solution counter.
    
    // Loop through n numbers.
    for (i = 0; i < n; i++) {
      scanf("%d ", &m);
      c = 0; // Reset lucky digit counter for the new number.
      
      // Count lucky digits in the number m.
      while (m != 0) {
        d = m % 10;
        m /= 10;
        if (d == 7 || d == 4) {
          c++;
        }
      }
      
      // If the count of lucky digits is within the limit, increment the solution counter.
      if (c <= k)
        s++;
    }
    
    // Print the total count of numbers with at most k lucky digits.
    printf("%d\n", s);
  }
  return 0;
}
