/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem on platforms like Codeforces.
 * @author Yadi Qian
 * @details This program simulates a simple programming language with one variable, X,
 * and two operations: "++" which increments X, and "--" which decrements X.
 * The program calculates the final value of X after a series of such operations.
 *
 * Algorithm:
 * 1. Initialize a variable 'value' (representing X) to 0.
 * 2. Read the number of operations, 'num'.
 * 3. Loop 'num' times:
 *    a. Read the operation as a string (e.g., "++X", "X--").
 *    b. If the string contains a '+', increment 'value'.
 *    c. Otherwise (if it contains a '-'), decrement 'value'.
 * 4. Print the final 'value'.
 *
 * Time Complexity: O(N), where N is the number of operations.
 * Space Complexity: O(1) (or O(S) if considering the string storage, where S is max operation length, which is constant).
 */
#include <bits/stdc++.h>

using namespace std;

// Commonly used constants and type definitions for competitive programming.
const double pi = acos(-1.0);
typedef pair<int, int> PII;
typedef vector<int> VI;
typedef vector<string> VS;
typedef vector<PII> VII;
typedef vector<VI> VVI;
typedef map<int, int> MPII;
typedef set<int> SETI;
typedef multiset<int> MSETI;
typedef long int int32;
typedef unsigned long int uint32;
typedef long long int int64;
typedef unsigned long long int uint64;

int main() {
  // Optimizes I/O operations for faster execution.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int num;
  cin >> num;
  int value = 0;

  /**
   * @brief Main loop to process each operation.
   * Pre-condition: 'num' holds the number of operations. 'value' is 0.
   * Invariant: After each iteration, 'value' is updated according to the operation.
   */
  for (int i = 0; i < num; i++) {
    string op;
    cin >> op;
    // Check for increment or decrement operation.
    // The check `op[0] == '+' || op[2] == '+'` covers all increment cases ("++X", "X++").
    if (op[0] == '+' || op[2] == '+')
      value++;
    else // Covers all decrement cases ("--X", "X--").
      value--;
  }

  // Output the final value of X.
  cout << value;

  return 0;
}