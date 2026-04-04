/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem on Codeforces.
 *
 * @details The problem determines a user's gender based on their username.
 * If the number of distinct characters in the username is odd, the user is
 * considered male. If it's even, the user is female.
 *
 * Algorithm:
 * 1. Read the username string.
 * 2. Use a frequency array `letters` of size 26 to count occurrences of each
 *    lowercase English letter.
 * 3. Iterate through the username and update the frequency of each character.
 * 4. Iterate through the frequency array and count how many characters appeared
 *    at least once. This gives the number of distinct characters.
 * 5. Check if the count of distinct characters is odd or even.
 * 6. Print "IGNORE HIM!" for an odd count (male) and "CHAT WITH HER!" for an
 *    even count (female).
 *
 * Time Complexity: O(L), where L is the length of the username string. The loops
 * run L times and then a constant 26 times.
 * Space Complexity: O(1), as the frequency array size is constant (26).
 */
#include <bits/stdc++.h>
using namespace std;
// A collection of macros for common operations in competitive programming.
#define length(a) (int)a.size()
#define all(a) a.begin(), a.end()
#define tolowercase(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define touppercase(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define lsb(x) (x & (-x))
#define ctoi(a) (int)a - 48
#define itoc(a) a + '0'
#define MOD 1000000007
#define ll long long

int main() {
  // Fast I/O setup.
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  string str;
  cin >> str;
  // Frequency array to store counts of each character 'a' through 'z'.
  int letters[26] = {0};
  /**
   * @brief Populate the frequency map based on characters in the input string.
   */
  for (int i = 0; i < str.size(); i++) {
    letters[str[i] - 'a']++;
  }
  int count = 0;
  /**
   * @brief Count the number of distinct characters.
   * Logic: Any character that appeared in the string will have a non-zero
   * count in the frequency array.
   */
  for (int i = 0; i < 26; i++) {
    if (letters[i])
      count++;
  }
  // Determine the output based on the parity of the distinct character count.
  if (count % 2) {
    cout << "IGNORE HIM!\n";
  } else
    cout << "CHAT WITH HER!\n";
}