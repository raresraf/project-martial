/**
 * @file 236A.CPP
 * @brief Solution to a programming problem determining a response based on the number of distinct characters in a string.
 * @details This program counts the number of unique characters in an input string (a username) and
 * outputs "CHAT WITH HER!" if the count is even, or "IGNORE HIM!" if the count is odd.
 * This implementation uses the C++ Standard Library's `std::set` to efficiently handle uniqueness.
 *
 * Algorithm:
 * 1. Read the input string `s`.
 * 2. Initialize a `std::set<char>` named `diffChar`. A set automatically handles uniqueness,
 *    storing only one instance of each character.
 * 3. Iterate through the input string and insert each character into the set. The set's properties
 *    ensure that duplicates are ignored.
 * 4. After iterating through the entire string, the size of the set (`diffChar.size()`) will be
 *    equal to the number of distinct characters.
 * 5. Use the ternary operator to check if the size is odd or even.
 *    - If `diffChar.size() % 2` is 1 (true), the user is considered male: "IGNORE HIM!".
 *    - If it is 0 (false), the user is considered female: "CHAT WITH HER!".
 *
 * Time Complexity: O(N log K), where N is the length of the string and K is the number of unique characters (at most 26 for lowercase English alphabet).
 * Each insertion into the set takes O(log K) time.
 * Space Complexity: O(K), for storing the unique characters in the set. In the worst case (all characters are unique), this could be O(N).
 */
#include <iostream>
#include <set>

using namespace std;

// Note: The following macro definition is syntactically incorrect and is not used in the program.
// It should be '#define ll long long' without the trailing semicolon.
#define long long ll;

string s;

void solve() {
  set<char> diffChar;
  /**
   * Block Logic: Iterate through the string to populate a set of unique characters.
   * Pre-condition: 's' contains the input string.
   * Invariant: 'diffChar' contains all unique characters encountered so far in the string.
   */
  for (int i = 0; i < s.length(); i++) {
    // The `if` check is redundant as std::set::insert will automatically handle duplicates.
    if (diffChar.count(s[i]) == 0)
      diffChar.insert(s[i]);
  }
  
  // Determine the output based on the parity of the number of distinct characters.
  string answer = diffChar.size() % 2 ? "IGNORE HIM!" : "CHAT WITH HER!";
  cout << answer;
}

int main() {
  cin >> s;
  solve();
}
