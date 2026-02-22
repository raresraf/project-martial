/**
 * @ea993d6f-87ec-4a46-b680-32691b05fcc9/546A.CPP
 * @brief Solution for a competitive programming problem involving calculating borrowed money for purchases.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of bananas increases arithmetically: the first costs 'k', the second '2k', up to 'wk'.
 * The soldier has 'n' dollars initially. The task is to find the deficit, or 0 if no borrowing is needed.
 * This solution uses a direct loop to sum costs and C-style I/O for potentially faster execution.
 *
 * Algorithm:
 * 1. Read the base cost of a banana 'k', the initial money 'n', and the number of bananas 'w'.
 * 2. Calculate the total cost of buying 'w' bananas by summing `k * i` for `i` from 1 to `w`.
 * 3. Compare the total cost with the initial money.
 * 4. If the total cost is less than or equal to the initial money, print 0 (no borrowing needed).
 * 5. Otherwise, calculate the difference (total cost - initial money) and print this amount (the amount to borrow).
 *
 * Time Complexity: O(W) due to the loop for summing costs, where W is the number of bananas.
 * Space Complexity: O(1) - uses a fixed amount of memory for variables.
 */

/// Functional Utility: Developer's personal signature/comment.
///        From dust I came and to dust I shall return
/**
                    Asadullah Bin Rahman
                          CSE-17
    HAJEE MOHAMMAD DANESH SCIENCE AND TECHNOLOGY UNIVERSITY
*/
#include <bits/stdc++.h> // Functional Utility: Includes most standard C++ libraries, common in competitive programming.
#define sf(f) scanf("%d", &f) // Functional Utility: Macro for simplified integer input using scanf.
#define pf(f) printf("%d\n", f) // Functional Utility: Macro for simplified integer output with newline using printf.
#define PI 3.1415926535897932385 // Functional Utility: Defines the mathematical constant PI.
#define size 3000000 // Functional Utility: Defines a large constant, possibly for array sizes.
#define mp make_pair // Functional Utility: Macro for std::make_pair.
#define pb push_back // Functional Utility: Macro for std::push_back.
// Functional Utility: Built-in functions for counting set bits (population count).
#define one(x) __builtin_popcount(x)     /// int
#define onel(x) __builtin_popcountl(x)   /// long int
#define onell(x) __builtin_popcountll(x) /// long long
using namespace std;
typedef long long ll; // Functional Utility: Defines 'll' as an alias for long long.

/// Functional Utility: Commented-out arrays for directional movements, common in grid-based problems.
/// int dx[]= {-1,-1,0,0,1,1};
/// int dy[]= {-1,0,-1,1,0,1};
/// int dx[]= {0,0,1,-1};                 /*4 side move*/
/// int dy[]= {-1,1,0,0};                 /*4 side move*/
/// int dx[]= {1,1,0,-1,-1,-1,0,1};       /*8 side move*/
/// int dy[]= {0,1,1,1,0,-1,-1,-1};       /*8 side move*/
/// int dx[]={1,1,2,2,-1,-1,-2,-2};       /*knight move*/
/// int dy[]={2,-2,1,-1,2,-2,1,-1};       /*knight move*/
/// Functional Utility: ASCII values for character types.
/// Always Remember   'A'=65,'Z'=90 'a'=97 'z'=122 '0'=48

int main() {
  int i, k, n, w, cost = 0, pay;
  // Functional Utility: Reads the base cost 'k'.
  sf(k);
  // Functional Utility: Reads the initial money 'n'.
  sf(n);
  // Functional Utility: Reads the number of bananas 'w'.
  sf(w);

  // Block Logic: Calculates the total cost of 'w' bananas.
  // Pre-condition: 'k', 'n', 'w' are initialized.
  // Invariant: 'cost' accumulates the total cost based on the arithmetic progression.
  for (i = 1; i <= w; i++) {
    cost += k * i; // Functional Utility: Adds the cost of the i-th banana to the total.
  }

  // Conditional Logic: Checks if the total cost is less than or equal to the initial money.
  if (cost <= n)
    pf(0); // Functional Utility: If enough money, prints 0 (no borrowing).
  else {
    pay = cost - n; // Functional Utility: Calculates the amount to borrow.
    pf(pay); // Functional Utility: Prints the amount to borrow.
  }

  return 0; // Functional Utility: Indicates successful program execution.
}
