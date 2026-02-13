/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem using a frequency counting method.
 * @details The program determines if a user is male or female based on whether the number
 * of distinct characters in their username is even or odd. This implementation uses a
 * frequency array (a vector) to efficiently count the distinct characters.
 *
 * @section algorithm Algorithm
 * 1. Read the username string, which consists of lowercase English letters.
 * 2. A frequency array `cnt` (a vector of 26 integers) is created and initialized to zeros.
 *    Each index from 0 to 25 corresponds to a letter of the alphabet.
 * 3. The code iterates through the input string. For each character, it increments the
 *    corresponding counter in the frequency array (e.g., for 'a', `cnt[0]` is incremented).
 * 4. After counting, it iterates through the frequency array and sums up the number of
 *    non-zero entries. This sum, `c`, represents the total count of distinct characters.
 * 5. The parity of `c` is checked. If it's odd, "IGNORE HIM!" is printed. If it's even,
 *    "CHAT WITH HER!" is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the string, as it involves a single pass over the string to build the frequency map. This is more efficient than an O(L log L) sorting approach.
 * - Space Complexity: O(1), as the frequency array has a fixed size (26), independent of the input string length.
 */
#include <bits/stdc++.h>
#define fs first
#define sc second
#define pi 3.141592653589793238462643383279502884197
//#define e 2.7182818284590452353602874713526624977572

typedef long long int ll;
typedef long double ld;
typedef std::vector<ll> vi;
typedef std::vector<bool> vb;
typedef std::vector<char> vc;
typedef std::pair<ll, ll> pii;
typedef std::vector<pii> vii;

using namespace std;
const ll mod = 1e9 + 7;

int main() {
  // Fast I/O configuration
  std::ios_base::sync_with_stdio(0);
  cin.tie(0);
  string s;
  cin >> s;
  // Frequency array to count occurrences of each character ('a' through 'z').
  vi cnt(26, 0);
  // Populate the frequency array.
  for (int i = 0; i < s.size(); ++i)
    cnt[s[i] - 'a']++;
  int c = 0; // Variable to store the count of distinct characters.
  // Iterate through the frequency array to find how many characters appeared.
  for (int i = 0; i < 26; ++i) {
    // The boolean expression (cnt[i] != 0) evaluates to 1 if the character
    // was present, and 0 otherwise. This sum gives the distinct count.
    c += (cnt[i] != 0);
  }
  // Check the parity of the distinct character count.
  if (c % 2)
    cout << "IGNORE HIM!\n";
  else
    cout << "CHAT WITH HER!\n";
  return 0;
}