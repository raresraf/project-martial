/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem (Codeforces 282A).
 * @author competitive-programmer
 *
 * @details This program simulates a simple programming language with one variable, X,
 * and two operations: increment (++) and decrement (--). Given a sequence of
 * operations, it calculates the final value of X, which starts at 0.
 *
 * Algorithm:
 * 1. Initialize a counter `cnt` (representing X) to 0.
 * 2. Read the number of operations, `n`.
 * 3. Loop `n` times. In each iteration:
 *    a. Read the operation as a string.
 *    b. Check if the string contains a '+' character. If it does, it's an
 *       increment operation. Otherwise, it's a decrement.
 *    c. Increment or decrement `cnt` accordingly.
 * 4. After the loop, print the final value of `cnt`.
 *
 * Time Complexity: O(n), where `n` is the number of operations, because we
 * process each operation once.
 * Space Complexity: O(1), aside from the storage for the input string in each loop.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// Unused competitive programming macros.
#define vc vector<int>
#define gcd(x, y) __gcd(x, y)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define MAX 1000001
#define inf 999999999
#define PI acos(-1)
#define MOD 1000000007

using namespace std;

int main() {
  int n;
  string s;
  
  // The `while` loop is a common C-style competitive programming pattern for
  // handling multiple test cases, though for this problem, n is given only once.
  while (scanf("%d", &n) == 1) {
    // `cnt` represents the variable X, initialized to 0.
    int cnt = 0;
    
    /**
     * Block Logic: This loop processes each of the 'n' operations.
     */
    for (int i = 0; i < n; i++) {
      cin >> s;
      /**
       * Decision Logic: Determines if the operation is an increment or decrement.
       * The condition checks for "X++" or "++X". If a '+' is present
       * in the first two characters, it's an increment. Otherwise, it must be
       * a decrement.
       */
      if ((s[0] == 'X' && s[1] == '+') || s[0] == '+') {
        cnt++;
      } else {
        cnt--;
      }
    }
    // Print the final value of X.
    cout << cnt << endl;
  }
  return 0;
}
