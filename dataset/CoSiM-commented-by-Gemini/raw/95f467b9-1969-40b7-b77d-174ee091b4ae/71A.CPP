/**
 * @file 71A.CPP
 * @brief Processes queries on a frequency map with custom string hashing.
 * @warning The filename `71A.CPP` is misleading. This code does not solve the
 *          "Way Too Long Words" problem (Codeforces 71A). It implements a
 *          solution for a query-based problem.
 *
 * @details
 * This program maintains a frequency map (using an array) where keys are derived
 * from strings. It reads `n` queries, each consisting of a type ('+', '-', or '?')
 * and a string. The string is converted into a numerical index using a custom
 * hashing function, and the corresponding value in the frequency map is either
 * incremented, decremented, or printed.
 *
 * @algorithm Custom Hashing + Frequency Map
 * @complexity
 * - `strToBin`: O(S), where S is the length of the string.
 * - `main`: O(N * S), where N is the number of queries.
 * - Space: O(MAX_A), where MAX_A is the size of the frequency array `a`.
 */
#include <bits/stdc++.h>
using namespace std;
#define pb push_back
#define ll long long

int n;
// `a` serves as a frequency map. The index is the hash of a string.
int a[300010];

/**
 * @brief Custom hash function to convert a string of digits into an integer.
 * @details This function treats the input string as a binary representation, but the
 * value of each bit is determined by the parity of the ASCII value of the
 * character digit. '1', '3', '5', '7', '9' are treated as binary '1', while
 * '0', '2', '4', '6', '8' are treated as binary '0'.
 * @param st The input string of digits.
 * @return A long long integer representing the calculated hash value.
 */
ll strToBin(string st) {
  ll res = 0;
  ll j = 1; // Represents the power of 2 for the current bit position.
  for (int i = st.length() - 1; i >= 0; i--) {
    // Check if the ASCII value of the character is odd.
    if (st[i] % 2 == 1) {
      res += j;
    }
    j *= 2;
  }
  return res;
}

int main() {
  cin >> n;
  /**
   * @brief Block Logic: Main loop to process `n` queries.
   */
  for (int i = 0; i < n; i++) {
    char c; // The query type: '+', '-', or '?'.
    string s;
    cin >> c >> s;
    
    // Hash the input string to get an index for the frequency array.
    ll num = strToBin(s);

    // Process the query based on its type.
    if (c == '?') {
      // Query: print the current count for this hash.
      cout << a[num] << endl;
    } else if (c == '+') {
      // Add: increment the count for this hash.
      a[num]++;
    } else if (c == '-') {
      // Remove: decrement the count for this hash.
      a[num]--;
    }
  }
  return 0;
}