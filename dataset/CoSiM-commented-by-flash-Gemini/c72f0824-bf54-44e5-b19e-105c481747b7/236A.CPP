/**
 * @file 236A.CPP
 * @brief This program determines if an input string contains an odd or even
 *        number of unique lowercase English letters. Based on the parity
 *        of unique characters, it outputs "CHAT WITH HER!" or "IGNORE HIM!".
 *        Two distinct approaches are presented: one using a frequency array
 *        and another utilizing `std::set`.
 * Algorithm: Counting Unique Characters (Frequency Array and Set-based).
 * Time Complexity: O(L) where L is the length of the input string for both approaches.
 *                  (Frequency array: linear scan; Set: linear scan + log(C) per insert, C unique chars).
 * Space Complexity: O(1) for `solve()` (fixed-size frequency array of 26).
 *                   O(C) for `solve2()` (std::set stores up to 26 unique characters).
 */
#include <bits/stdc++.h>
using namespace std;

#define endl "\n"
#define ends return 0;
#define fast                                                                   \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);

/**
 * @brief Optimizes C++ standard input/output operations for faster execution.
 * Functional Utility: Disables synchronization with C's stdio and unties cin from cout,
 *                     which is beneficial for competitive programming.
 */
void fast() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
}

// 1st attempt
/**
 * @brief Solves the problem using a frequency array to count unique characters.
 * Functional Utility: Counts unique lowercase English letters in a string
 *                     and determines if the count is odd or even.
 */
void solve() {
  string s; // Input string.
  getline(cin, s); // Reads the entire line as input.
  int cnt = 0; // Counter for unique characters.

  vector<int> v(26, 0); // Frequency array for 26 lowercase English letters, initialized to 0.

  /**
   * Block Logic: Iterates through the input string to mark and count unique characters.
   * Invariant: `v[x]` is set to 1 if the character corresponding to `x` has been encountered,
   *            and `cnt` is incremented only once for each unique character.
   */
  for (int i = 0; s[i]; i++) {
    int x = s[i] - 'a'; // Converts character to 0-25 index (e.g., 'a' -> 0).
    if (v[x] == 0) {    // If this character has not been seen before.
      v[x] = 1;         // Mark it as seen.
      cnt++;            // Increment unique character count.
    }
  }

  // Determines output based on parity of unique character count.
  if (cnt % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}

// Second attempt(using stl)
/**
 * @brief Solves the problem using `std::set` to count unique characters.
 * Functional Utility: Leverages the `std::set` container's property of
 *                     storing only unique elements to efficiently count unique characters.
 */
void solve2() {
  string s; // Input string.
  getline(cin, s); // Reads the entire line as input.
  int size = s.length(); // Length of the input string.
  set<char> S;           // `std::set` to store unique characters.

  /**
   * Block Logic: Inserts each character of the string into the set.
   * Invariant: After the loop, `S` contains all unique characters from the string.
   */
  for (int i = 0; i < size; i++) {
    S.insert(s[i]); // `std::set` automatically handles uniqueness.
  }

  // Determines output based on parity of unique character count (set size).
  if (S.size() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}

/**
 * @brief Main function: initializes fast I/O and calls the `solve2()` function.
 * Functional Utility: Entry point of the program, orchestrates the problem solution.
 */
int main() {
  fast;   // Calls the fast I/O setup function.
  solve2(); // Executes the second (set-based) solution.
  ends;   // Equivalent to return 0;.
}