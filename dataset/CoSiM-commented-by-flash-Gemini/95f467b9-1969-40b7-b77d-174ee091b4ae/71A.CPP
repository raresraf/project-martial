/**
 * @file 71A.CPP
 * @brief Processes operations on binary strings, mapping them to numerical indices for frequency tracking.
 * * Algorithm: Binary String to Integer Conversion with Frequency Array Operations.
 * Time Complexity: $O(N \times L)$ where N is the number of operations and L is the average length of binary strings.
 * Space Complexity: $O(M)$ where M is the maximum possible numerical value of a binary string, for the frequency array.
 *
 * This program reads a series of operations (+, -, ?) on binary strings. Each binary string
 * is first converted to an integer index. The operations then modify or query a global
 * frequency array based on this index. This effectively allows tracking counts associated
 * with specific binary patterns.
 */

#include <bits/stdc++.h>
using namespace std;

// Macros and type definitions common in competitive programming.
#define pb push_back // Shorthand for vector::push_back.
#define ll long long // Alias for long long integer type.

int n; // Number of operations to process.
// Global array `a` to store frequencies or counts. Its size (300010) implies it can handle
// a range of numerical values derived from the binary strings.
int a[300010]; 

/**
 * @brief Converts a binary string (composed of '0's and '1's) into its decimal `long long` equivalent.
 *
 * This function processes the string from right to left, assigning powers of 2
 * to each '1' character. It effectively treats the input as a binary number.
 *
 * @param st The input binary string (e.g., "101", "1100").
 * @return The decimal `long long` representation of the binary string.
 */
ll strToBin(string st) {
  ll res = 0; // Accumulator for the decimal result.
  ll j = 1;   // Represents the current power of 2 (1, 2, 4, 8, ...).
  // Block Logic: Iterates through the binary string from the least significant bit (rightmost character)
  // to the most significant bit (leftmost character).
  // Invariant: `res` accumulates the decimal value, and `j` tracks the current power of 2.
  for (int i = st.length() - 1; i >= 0; i--) {
    // Inline: Checks if the character is '1'. In ASCII, '1' is odd, '0' is even.
    // So, `st[i] % 2 == 1` effectively checks if `st[i]` is '1'.
    if (st[i] % 2 == 1) { 
      res += j; // If '1', add the current power of 2 to the result.
    }
    j *= 2; // Move to the next power of 2 for the next bit position.
  }
  return res; // Returns the final decimal value.
}

int main() {
  cin >> n; // Reads the total number of operations.
  // Block Logic: Processes `n` operations, each consisting of a command character and a binary string.
  // Invariant: After each iteration, one operation is performed on the frequency array `a`.
  for (int i = 0; i < n; i++) {
    char c;    // Operation character ('+', '-', '?').
    string s;  // Binary string for the operation.
    cin >> c >> s; // Reads the operation character and the binary string.
    
    // Converts the binary string to its numerical index.
    ll num = strToBin(s);
    
    // Block Logic: Executes the operation based on the character 'c'.
    if (c == '?') {
      // Inline: Query operation: Prints the frequency count for the calculated binary pattern.
      cout << a[num] << endl;
    } else if (c == '+') {
      // Inline: Increment operation: Increases the frequency count for the binary pattern.
      a[num]++;
    } else if (c == '-') {
      // Inline: Decrement operation: Decreases the frequency count for the binary pattern.
      a[num]--;
    }
  }
  return 0; // Standard successful exit code.
}
