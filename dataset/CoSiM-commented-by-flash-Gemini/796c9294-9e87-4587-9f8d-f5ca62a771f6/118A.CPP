/**
 * @file 118A.CPP
 * @brief Solution for Codeforces Problem (implied): Check if two strings can be made equal by one swap.
 * @details This program takes two strings as input and determines if they can be made identical
 * by swapping exactly two characters. It identifies all positions where the characters differ,
 * and then checks if there are exactly two such positions and if the characters at these positions,
 * when swapped, would make the strings equal.
 * @algorithm String comparison, difference tracking, and swap validation.
 * Time Complexity: $O(L)$ where L is the length of the strings.
 * Space Complexity: $O(1)$ or $O(L)$ in worst case for storing differing indices, depending on implementation detail (here, `st` vector).
 */
#include <bits/stdc++.h>
using namespace std;
#pragma GCC optimize("-O3")

#define ll long long int
#define pll pair<ll, ll>
#define pls pair<ll, string>
#define psl pair<string, ll>
#define plc pair<ll, char>
#define pcl pair<char, ll>
#define pss pair<string, string>
#define maxheap(type) priority_queue<type>
#define minheap(type) priority_queue<type, vector<type>, greater<type>>
#define setbits(x) __builtin_popcountll(x)
#define zerobits(x) __builtin_ctzll(x)
#define mod 1000000007
#define inf 1e18
#define precise(x, y) fixed << setprecision(y) << x
#define tol(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define tou(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define rep(n) for (auto i = 0; i < n; i++)

/**

 * @file 118A.CPP

 * @brief Solution for Codeforces Problem (implied): Check if two strings can be made equal by one swap.

 * @details This program takes two strings as input and determines if they can be made identical

 * by swapping exactly two characters. It identifies all positions where the characters differ,

 * and then checks if there are exactly two such positions and if the characters at these positions,

 * when swapped, would make the strings equal.

 * @algorithm String comparison, difference tracking, and swap validation.

 * Time Complexity: $O(L)$ where L is the length of the strings.

 * Space Complexity: $O(1)$ or $O(L)$ in worst case for storing differing indices, depending on implementation detail (here, `st` vector).

 */

#include <bits/stdc++.h> // Includes most standard C++ library headers.

using namespace std;     // Uses the standard namespace.



#pragma GCC optimize("-O3") // Compiler optimization directive for GCC.



// Define useful type aliases for competitive programming.

#define ll long long int           // Alias for long long integer.

#define pll pair<ll, ll>           // Alias for pair of long long integers.

#define pls pair<ll, string>       // Alias for pair of long long and string.

#define psl pair<string, ll>       // Alias for pair of string and long long.

#define plc pair<ll, char>         // Alias for pair of long long and char.

#define pcl pair<char, ll>         // Alias for pair of char and long long.

#define pss pair<string, string>   // Alias for pair of strings.



// Define heap-related macros.

#define maxheap(type) priority_queue<type>                         // Max-heap definition.

#define minheap(type) priority_queue<type, vector<type>, greater<type>> // Min-heap definition.



// Define bit manipulation macros.

#define setbits(x) __builtin_popcountll(x) // Counts set bits (1s) in a long long.

#define zerobits(x) __builtin_ctzll(x)     // Counts trailing zeros in a long long.



// Define common constants.

#define mod 1000000007 // Modulo for arithmetic operations.

#define inf 1e18       // Represents infinity for comparisons.



// Define formatting macros.

#define precise(x, y) fixed << setprecision(y) << x // Sets precision for floating-point output.



// Define string transformation macros.

#define tol(s) transform(s.begin(), s.end(), s.begin(), ::tolower); // Converts string to lowercase.

#define tou(s) transform(s.begin(), s.end(), s.begin(), ::toupper); // Converts string to uppercase.



// Define a common loop macro.

#define rep(n) for (auto i = 0; i < n; i++) // Shorthand for a common for loop.



int main() {

  /**

   * @brief Main function: orchestrates string comparison and validation logic.

   */



  // Functional Utility: Optimizes C++ standard streams for competitive programming.

  // Disables synchronization with C's stdio and unties cin from cout.

  ios_base::sync_with_stdio(0);

  cin.tie(NULL);

  cout.tie(NULL);



  string s1, s2; // @var s1, s2: The two input strings to be compared.

  cin >> s1 >> s2; // Reads the two strings from standard input.



  /**

   * @brief Initial check for string length equality.

   * @block_logic If strings have different lengths, they cannot be made equal by swapping characters.

   * @invariant Returns "NO" and exits if lengths differ.

   */

  if (s1.size() != s2.size()) {

    cout << "NO\n";

    return 0;

  }



  ll n = s1.size(); // @var n: Stores the length of the strings.

  vector<ll> st;    // @var st: Vector to store the indices where s1 and s2 differ.



  /**

   * @brief Iterates through strings to find differing characters and track their indices.

   * @block_logic Compares characters at each position. If different, adds index to `st`.

   *              Also checks if more than two differences are found prematurely.

   * @invariant `st` contains indices of differing characters. If `st.size()` exceeds 2, condition is false.

   */

  rep(n) { // Loop from `i = 0` to `n - 1`.

    auto ch1 = s1[i], ch2 = s2[i]; // @var ch1, ch2: Characters at current index `i` from `s1` and `s2`.

    if (ch1 != ch2) { // If characters at current position differ.

      st.push_back(i); // Store the index of difference.

    }

    /**

     * @brief Checks if the number of differing characters exceeds two.

     * @block_logic If more than two differences are found, it's impossible to fix with one swap.

     * @invariant Returns "NO" and exits if `st.size()` becomes greater than 2.

     */

    if (st.size() > 2) {

      cout << "NO\n";

      return 0;

    }

  }



  /**

   * @brief Final validation: checks if exactly two differences exist and if swapping them makes strings equal.

   * @block_logic Checks two conditions:

   *              1. `st.size() == 2`: Exactly two positions must differ.

   *              2. `s1[st[0]] == s2[st[1]] && s1[st[1]] == s2[st[0]]`: The characters at the first differing

   *                 position in `s1` must match the character at the second differing position in `s2`,

   *                 and vice-versa, to simulate a successful swap.

   * @invariant Returns "YES" if both conditions are met, otherwise "NO".

   */

  if ((st.size() != 2) || // Check if exactly two differences were found.

      !((s1[st[0]] == s2[st[1]]) && (s1[st[1]] == s2[st[0]]))) { // Check if characters at differing positions are swappable.

    cout << "NO\n";

    return 0;

  }



  cout << "YES\n"; // If all checks pass, strings can be made equal by one swap.

  return 0;        // Indicates successful execution.

}
