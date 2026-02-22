/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A: Way Too Long Words.
 * @details This program processes a list of words and abbreviates those that exceed a certain length (10 characters).
 * For words longer than 10 characters, the abbreviation consists of the first letter, followed by the count
 * of characters between the first and last letters, and then the last letter. Words 10 characters or shorter
 * are printed as they are. This implementation uses various competitive programming optimizations and macros.
 * @algorithm String manipulation and conditional formatting.
 * Time Complexity: $O(L_{avg} \cdot N)$ where N is the number of words and $L_{avg}$ is the average length of words.
 * Space Complexity: $O(L_{max})$ to store the longest word, where $L_{max}$ is the maximum word length.
 * @author Anmol Gupta
 */
#include <bits/stdc++.h>
using namespace std;
#pragma GCC optimize("O3")
#define pb push_back
#define debug(x) cout << '>' << #x << " : " << x << endl;
#define mp make_pair
#define ff first
#define ss second
#define allunique(v) v.erase(unique(v.begin(), v.end()), v.end());
#define endl "\n"
#define MOD 1000000007
#define MOD9 1000000009
#define pin(n) cout << n << endl
#define gc getchar_unlocked
#define ms(s, n) memset(s, n, sizeof(s))
#define prec(n) fixed << setprecision(n)
#define eps 0.000001
#define pll pair<ll, ll>
#define all(v) v.begin(), v.end()
#define bolt ios::sync_with_stdio(0)
#define forr(i, n) for (ll i = 0; i < n; i++)
#define eof (scanf("%d", &n)) != EOF
#define PI acos(-1.0)
typedef long long ll;

/**

 * @file 71A.CPP

 * @brief Solution for Codeforces Problem 71A: Way Too Long Words.

 * @details This program processes a list of words and abbreviates those that exceed a certain length (10 characters).

 * For words longer than 10 characters, the abbreviation consists of the first letter, followed by the count

 * of characters between the first and last letters, and then the last letter. Words 10 characters or shorter

 * are printed as they are. This implementation uses various competitive programming optimizations and macros.

 * @algorithm String manipulation and conditional formatting.

 * Time Complexity: $O(L_{avg} \cdot N)$ where N is the number of words and $L_{avg}$ is the average length of words.

 * Space Complexity: $O(L_{max})$ to store the longest word, where $L_{max}$ is the maximum word length.

 * @author Anmol Gupta

 */

#include <bits/stdc++.h> // Includes most standard C++ library headers (e.g., iostream, string, algorithm).

using namespace std;     // Uses the standard namespace.



#pragma GCC optimize("O3") // @pragma GCC optimize: Enables aggressive GCC optimizations for performance.



// Define macro aliases for common competitive programming tasks and types.

#define pb push_back                                                       // @def pb: Shorthand for std::vector::push_back.

#define debug(x) cout << '>' << #x << " : " << x << endl;                 // @def debug: Prints variable name and value for debugging.

#define mp make_pair                                                       // @def mp: Shorthand for std::make_pair.

#define ff first                                                           // @def ff: Shorthand for accessing the first element of a pair.

#define ss second                                                          // @def ss: Shorthand for accessing the second element of a pair.

#define allunique(v) v.erase(unique(v.begin(), v.end()), v.end());         // @def allunique: Removes duplicate elements from a sorted vector.

#define endl "\n"                                                          // @def endl: Uses newline character instead of `std::endl` for faster output.

#define MOD 1000000007                                                     // @def MOD: A common modulo value for arithmetic operations.

#define MOD9 1000000009                                                    // @def MOD9: Another common modulo value.

#define pin(n) cout << n << endl                                           // @def pin: Prints an integer followed by a newline.

#define gc getchar_unlocked                                                // @def gc: Faster input character function (non-standard).

#define ms(s, n) memset(s, n, sizeof(s))                                   // @def ms: Sets a block of memory to a specified value.

#define prec(n) fixed << setprecision(n)                                   // @def prec: Sets output precision for floating-point numbers.

#define eps 0.000001                                                       // @def eps: A small epsilon value for floating-point comparisons.

#define pll pair<ll, ll>                                                   // @def pll: Alias for pair of long long integers.

#define all(v) v.begin(), v.end()                                          // @def all: Returns iterators to the beginning and end of a container.

#define bolt ios::sync_with_stdio(0)                                       // @def bolt: Disables synchronization with C stdio for faster I/O.

#define forr(i, n) for (ll i = 0; i < n; i++)                              // @def forr: Shorthand for a common for loop with long long index.

#define eof (scanf("%d", &n)) != EOF                                       // @def eof: Checks for End-Of-File after scanf (potentially captures an int `n`).

#define PI acos(-1.0)                                                      // @def PI: Defines the mathematical constant PI.

typedef long long ll; // @typedef ll: Alias for long long integer. Redundant with `#define ll long long`.



int main() {

  /**

   * @brief Main function: orchestrates the word abbreviation logic.

   */



  // Functional Utility: Optimizes C++ standard streams for competitive programming.

  // Disables synchronization with C's stdio and unties cin from cout.

  ios_base::sync_with_stdio(false);

  cin.tie(NULL);

  cout.tie(NULL);



  // The commented line `// printf("%I64d", n)` appears to be a leftover from debugging or boilerplate.



  ll t; // @var t: The total number of words to process.

  cin >> t; // Reads the total number of words.



  /**

   * @brief Processes each of the 't' words.

   * @block_logic Iterates `t` times, processing one word in each iteration.

   * @invariant `t` decreases by 1 in each iteration until it becomes 0.

   */

  while (t--) {

    string str; // @var str: Stores the current word read from input.

    cin >> str; // Reads the current word.

    int len = str.length(); // @var len: Calculates the length of the current word.



    /**

     * @brief Checks if the word needs abbreviation.

     * @block_logic If `len` is greater than 10, abbreviates the word; otherwise, prints it as is.

     * @invariant Words are either abbreviated or printed in full based on their length.

     */

    if (len > 10) {

      // Abbreviate the word: first character + (length - 2) + last character.

      cout << str[0]       // First character.

           << len - 2      // Number of characters between first and last.

           << str[len - 1] // Last character.

           << endl;        // Newline.

    } else {

      // Print the word as is if its length is 10 or less.

      cout << str << endl; // The original word followed by a newline.

    }

  }

}