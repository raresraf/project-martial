/**
 * @file 236A.CPP
 * @brief Solution for Codeforces Problem 236A - "Boy or Girl".
 * @details The program determines a user's gender based on the number of distinct characters
 * in their name. If the count of distinct characters is odd, the user is male ("IGNORE HIM!").
 * If the count is even, the user is female ("CHAT WITH HER!").
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. An array `alphabet` of size 29 is used as a frequency map or a set to track the
 *    presence of each character ('a' through 'z').
 * 2. The program reads the user's name as a string.
 * 3. It iterates through the name, and for each character, it marks the corresponding
 *    index in the `alphabet` array (e.g., 'a' maps to index 0, 'b' to 1, etc.).
 * 4. Instead of counting the number of distinct characters directly, the code counts the
 *    number of characters *not* present in the name (`cnt`).
 * 5. The number of distinct characters is therefore `29 - cnt`. The program then checks
 *    the parity of `cnt`. Checking if `cnt` is odd/even is equivalent to checking if
 *    `(29 - num_distinct_chars)` is odd/even, which correctly determines the output.
 *    - If (29 - d) is odd (meaning d is even), output "CHAT WITH HER!".
 *    - If (29 - d) is even (meaning d is odd), output "IGNORE HIM!".
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the name, for iterating through the string.
 * The subsequent loop is constant time (29 iterations).
 * Space Complexity: O(1), for the fixed-size alphabet array.
 */
#include <bits/stdc++.h>
// Collection of macros for competitive programming to shorten common code patterns.
#define nl '\n'
#define ll long long int
#define loop(i, a, b) for (int i = a; i < b; ++i)
#define count_1(n) __builtin_popcountll(n)
#define pb push_back
#define F first
#define S second
#define mp make_pair
#define clr(x) x.clear()
#define MOD 1000000007
#define itoc(c) ((char)(((int)'0') + c))
#define vi vector<int>
#define pll pair<ll, ll>
#define pii pair<int, int>
#define all(p) p.begin(), p.end()
#define max(x, y) (x > y) ? x : y
#define min(x, y) (x < y) ? x : y
#define mid(s, e) (s + (e - s) / 2)
#define tcase()                                                                \
  ll t;                                                                        \
  scanf("%lld", &t);                                                           \
  while (t--)
#define iscn(num) scanf("%d", &num);
using namespace std;
void FAST_IO();
// A boolean array to act as a set for the characters 'a' through 'z'.
// Size 29 is slightly larger than 26, providing a buffer.
bool alphabet[29];
int main() {
  FAST_IO();
  string name;
  cin >> name;
  int cnt = 0;
  /**
   * Block Logic: This loop populates the `alphabet` array to mark which characters
   * are present in the input `name`. This effectively creates a set of unique characters.
   */
  for (int i = 0; i < name.size(); i++)
    alphabet[name[i] - 'a'] = 1;
  /**
   * Block Logic: This loop counts the number of characters in the alphabet that were
   * *not* found in the name.
   */
  for (int i = 0; i < 29; i++) {
    if (!alphabet[i])
      cnt++;
  }
  /**
   * Block Logic: Determines the output based on the parity of the count of *missing* characters.
   * If the number of distinct characters `d` is even, `29 - d` (`cnt`) will be odd.
   * If `d` is odd, `29 - d` (`cnt`) will be even.
   * The logic correctly maps the parity of distinct characters to the required output.
   */
  if (cnt % 2) // True if cnt is odd, which means the number of distinct characters is even.
    cout << "CHAT WITH HER!" << nl;
  else // cnt is even, which means the number of distinct characters is odd.
    cout << "IGNORE HIM!" << endl;
  return 0;
}

/**
 * @brief A utility function to speed up C++ stream I/O.
 * This is a standard optimization in competitive programming.
 */
void FAST_IO() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
// cout.setf(ios::fixed);
// cout.precision(20);
#ifndef _offline
// These lines are typically used for local testing to redirect I/O to files.
// freopen("input.txt", "r", stdin);
// freopen("output.txt", "w", stdout);
#endif
}
