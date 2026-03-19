/**
 * @file raw/c72f0824-bf54-44e5-b19e-105c481747b7/236A.CPP
 * @brief Contains two solutions for determining an action based on the number
 *        of distinct characters in a username.
 *
 * This file presents two different approaches to solving the "Boy or Girl" problem,
 * where the output depends on whether the count of unique characters in a string
 * is even or odd. The main function exclusively calls the second, more idiomatic
 * STL-based solution.
 */
#include <bits/stdc++.h>
using namespace std;

// --- Competitive Programming Boilerplate ---
#define endl "
"
#define ends return 0;
#define fast                                                                   
  ios::sync_with_stdio(0);                                                     
  cin.tie(0);                                                                  
  cout.tie(0);

/**
 * @brief First attempt at solving the problem using a manual frequency map.
 * @note This function is defined but not called by main().
 *
 * @algorithm Uses a vector of size 26 as a frequency map to track which
 *            characters have been seen. It iterates through the string, and for
 *            each new character, it increments a counter.
 * @time_complexity O(L), where L is the string length.
 * @space_complexity O(1), for the fixed-size vector (26).
 */
void solve() {
  string s;
  getline(cin, s);
  int cnt = 0;

  vector<int> v(26, 0); // Serves as a hash set for the alphabet.

  for (int i = 0; s[i]; i++) {
    int x = s[i] - 'a';
    if (v[x] == 0) { // If character is new
      v[x] = 1;      // Mark as seen
      cnt++;         // Increment unique count
    }
  }

  if (cnt % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}

/**
 * @brief Second, active solution using the C++ Standard Template Library (STL).
 *
 * @algorithm This function leverages `std::set`, a container that automatically
 *            stores only unique elements. By inserting every character from the
 *            string into the set, the final size of the set (`S.size()`) gives
 *            the exact count of distinct characters. This is a cleaner and more
 *            robust approach than manual counting.
 * @time_complexity O(L log K), where L is the string length and K is the number
 *                  of unique characters (K <= 26). Insertion into a set has
 *                  logarithmic complexity.
 * @space_complexity O(K) to store the unique characters in the set.
 */
void solve2() {
  string s;
  getline(cin, s);
  int size = s.length();
  set<char> S;

  // Insert all characters into the set. Duplicates are automatically ignored.
  for (int i = 0; i < size; i++) {
    S.insert(s[i]);
  }

  // The size of the set is the number of distinct characters.
  if (S.size() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}
int main() {
  fast;
  // The program calls the second, more idiomatic STL-based solution.
  solve2();
  ends;
}
