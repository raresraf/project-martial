/**
 * @file 282A.CPP
 * @brief Solution to Codeforces problem 282A - "Bit++".
 * @details This program simulates a series of increment and decrement operations on a single variable and prints its final value.
 * The operations are given as strings, and the logic checks for the presence of '+' or '-' characters to determine the operation type.
 *
 * Algorithm:
 * 1. Initialize a counter to zero.
 * 2. Read the total number of operations, `n`.
 * 3. Loop `n` times:
 *    a. Read the operation string.
 *    b. If the string contains a '-', decrement the counter.
 *    c. If the string contains a '+', increment the counter.
 * 4. Print the final value of the counter.
 *
 * Time Complexity: O(N), where N is the number of operations.
 * Space Complexity: O(1), as storage is constant regardless of input size (excluding input string storage).
 */

/*==============================================*
Codeforces ID:    mdshihab
                                                  |
Name:     Md. Shihabul Minhaz                     |
Study:      CSE, JKKNIU                           |
Address:  Trishal, Mymensingh, Bangladesh         |
                                                  |
 mail:   mdshihabul20171@gmail.com                |
 FB  :   fb.com/mdshihabul.minhaz.7               |
 github: Shihabulminhaz                           |
stopstalk :  mdshihab                             |
                                                  |
@uthor   Md. Shihabul Minhaz (shihab)             |
\*===============================================*/

#include <bits/stdc++.h>
using namespace std;
#define fi(a) scanf("%d", &a);
#define fli(a) scanf("%ld", &a);
#define fll(a) scanf("%lld", &a);
#define pi(a) printf("%d
", a);
#define ppi(i, a) printf("Case %d: %d
", i, a);
#define ll long long
#define fast                                                                   
  ios_base::sync_with_stdio(false);                                            
  cin.tie(NULL);                                                               
  cout.tie(NULL);
void FI() {
#ifndef ONLINE_JUDGE
  freopen("C:\Users\SHIHAB\Desktop\input.in", "r", stdin);
#endif // ONLINE_JUDGE
}

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and flushes
   * to optimize high-throughput I/O, a common practice in competitive programming.
   */
  fast
      // FI(); /// skip this line for compile
      int n,
      cnt = 0;
  string s;
  cin >> n;

  /**
   * @brief Main loop to process `n` operations.
   * Invariant: `cnt` holds the cumulative value of the simulated variable after each operation.
   */
  while (n--) {
    cin >> s;
    /**
     * @brief Checks the operation type based on character presence.
     * The problem statement guarantees that the operation string will be one of "++X", "X++", "--X", or "X--".
     * Checking the first or third character for '-' is sufficient to detect a decrement operation.
     */
    if (s[0] == '-' || s[2] == '-')
      cnt--;
    else
      cnt++;
  }
  
  // Output the final value of the counter after all operations.
  cout << cnt << endl;
  return 0;
}
