/**
 * @file 282A.CPP
 * @brief Simulation of a minimalist state machine for processing bitwise-style instructions.
 * 
 * Logic: Tracks the state of a single numeric register (cnt) across a series 
 * of operations. Commands containing '-' are interpreted as decrements, 
 * while others (by implication '++') are increments. Supports both prefix 
 * and postfix notations by checking boundary characters of the command string.
 * 
 * Algorithm: Pattern matching on command strings to update a numeric accumulator.
 * Time Complexity: $O(N)$ where N is the number of instructions.
 * Space Complexity: $O(1)$ constant auxiliary space beyond the small command buffer.
 */

#include <bits/stdc++.h>
using namespace std;

// Functional Utility: Macro-based IO and redirection for competitive programming environments.
#define fi(a) scanf("%d", &a);
#define fli(a) scanf("%ld", &a);
#define fll(a) scanf("%lld", &a);
#define pi(a) printf("%d\n", a);
#define ppi(i, a) printf("Case %d: %d\n", i, a);
#define ll long long

/**
 * Functional Utility: Optimizes standard I/O throughput by decoupling from C standard streams.
 */
#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);

/**
 * @brief Configures local file redirection for development debugging.
 */
void FI() {
#ifndef ONLINE_JUDGE
  freopen("C:\\Users\\SHIHAB\\Desktop\\input.in", "r", stdin);
#endif // ONLINE_JUDGE
}

int main() {
  fast
  
  int n, cnt = 0;
  string s;
  
  // Reads the total number of operations to be executed.
  if (!(cin >> n)) return 0;

  /**
   * Block Logic: Main instruction decoding and execution loop.
   * Invariant: Each command is mapped to exactly one state transition (+1 or -1).
   */
  while (n--) {
    // Reads an instruction string (e.g., "X++", "--X", "++X", "X--").
    cin >> s;
    
    /**
     * Functional Utility: Pattern matching for operation decoding.
     * Logic: If either the first or the third character is '-', the operation is a decrement.
     * This correctly handles both prefix ("--X") and postfix ("X--") decrements.
     */
    if (s[0] == '-' || s[2] == '-')
      cnt--;
    else
      // Implicitly handles "++X" and "X++".
      cnt++;
  }
  
  // Output the final state of the virtual register.
  cout << cnt << endl;
  
  return 0;
}
