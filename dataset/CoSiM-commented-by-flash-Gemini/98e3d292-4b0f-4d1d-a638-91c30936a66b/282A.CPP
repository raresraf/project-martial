/**
 * @file 282A.CPP
 * @brief This program simulates a simple counter variable `x` that is incremented
 * or decremented based on a series of input operations. Each operation is a string;
 * if it contains "++", `x` is incremented, otherwise `x` is decremented.
 *
 * Algorithm: Iterative processing of operations.
 * 1. Initialize `x` to 0.
 * 2. Read the number of operations `n`.
 * 3. Loop `n` times:
 *    a. Read an operation string.
 *    b. Check if "++" is present in the string.
 *    c. Increment `x` if "++" is found, decrement otherwise.
 * 4. Print the final value of `x`.
 * Time Complexity: O(N * L) where N is the number of operations and L is the maximum length of an operation string,
 *                  due to string searching within the loop.
 * Space Complexity: O(L) to store the operation string.
 */

/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
typedef vector<int> vi;
typedef vector<long long> vl;

#define MOD 1000000007
#define _for(i, n) for (int i = 0; i < n; i++)
#define __for(i, n) for (int i = 1; i <= n; i++)
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i--)
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
#define tr(container, it)                                                      
  for (typeof(container.begin()) it = container.begin();                       
       it != container.end(); it++)
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define ALL(x) x.begin(), x.end()

/**
 * @brief Main function of the program.
 * Processes a sequence of operations to calculate the final value of a variable `x`.
 * @pre Input consists of an integer `n` followed by `n` operation strings.
 * @post Prints the final integer value of `x`.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: Reads the number of operations.
  int n;
  scanf("%d", &n);
  // Functional Utility: Stores the current operation string.
  string str;
  // Functional Utility: The counter variable, initialized to 0.
  int x = 0;
  // Block Logic: Iterates through each of the `n` operations.
  // Invariant: `x` correctly reflects the accumulated result of operations processed so far.
  for (int i = 0; i < n; i++) {
    // Functional Utility: Reads the operation string.
    cin >> str;
    // Block Logic: Checks if the operation string contains "++" to determine increment or decrement.
    // Pre-condition: `str` is the current operation string.
    if (str.find("++") != string::npos)
      x++; // Functional Utility: Increments `x`.
    else
      x--; // Functional Utility: Decrements `x`.
  }
  // Functional Utility: Prints the final value of `x`.
  printf("%d
", x);
  return 0;
}
