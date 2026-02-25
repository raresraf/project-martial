/**
 * @file 546A.CPP
 * @brief This program calculates the additional money required to buy a certain number of items,
 * where the cost of each successive item increases linearly.
 * Specifically, the first item costs 'n', the second '2n', the third '3n', and so on, up to 'w' items.
 * The user starts with 'k' dollars, and the program determines how much more money is needed.
 * Algorithm: Summation of an arithmetic progression.
 * Time Complexity: O(1) due to direct arithmetic calculations.
 * Space Complexity: O(1) for auxiliary variables.
 */
#include <iostream> // Includes the standard input/output stream library.
using namespace std;

#define end return 0 // Macro to use 'end' as an alias for 'return 0', a common competitive programming shorthand.

typedef long long ll; // Defines 'll' as a shorthand for 'long long'.
typedef long long int lli; // Defines 'lli' as a shorthand for 'long long int'.
typedef unsigned long long int ulli; // Defines 'ulli' as a shorthand for 'unsigned long long int'.

int main(int argc, char const *argv[]) {
  // Functional Utility: Disables synchronization between C++ standard streams and C standard streams.
  // This can significantly speed up I/O operations in competitive programming.
  ios_base::sync_with_stdio(0);
  lli n, k, w, a, b, c; // Declares variables: n (base cost), k (initial money), w (number of items), a (total cost), b (money needed), c (unused).
  cin >> n >> k >> w; // Reads the input values for base cost, initial money, and number of items.
  /**
   * Functional Utility: Calculates the total cost 'a' of buying 'w' items.
   * The cost of the i-th item is 'i * n'.
   * The total cost is n * (1 + 2 + ... + w) = n * w * (w + 1) / 2.
   * This is the sum of an arithmetic progression.
   */
  a = (w * (w + 1) / 2) * n;
  /**
   * Functional Utility: Calculates the additional money 'b' needed.
   * This is the total cost 'a' minus the initial money 'k'.
   */
  b = a - k;
  // Conditional Logic: Checks if the calculated money needed 'b' is negative.
  // If 'b' is negative, it means the user has enough or more money than required.
  if (a - k < 0) {
    cout << 0 << endl; // Prints 0, indicating no additional money is needed.
  } else {
    cout << b << endl; // Prints 'b', the exact amount of money needed.
  }
  end; // Uses the 'end' macro, which expands to 'return 0'.
}
