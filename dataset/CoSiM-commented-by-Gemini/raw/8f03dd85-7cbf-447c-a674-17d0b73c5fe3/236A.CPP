/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem (e.g., Codeforces 236A).
 * @details This program determines a user's gender based on the number of distinct
 * characters in their username. If the count of distinct characters is odd, the
 * user is identified as male ("IGNORE HIM!"). If the count is even, the user is
 * identified as female ("CHAT WITH HER!").
 *
 * @algorithm Counting distinct elements using a std::set.
 * @complexity
 * - Time: O(L), where L is the length of the input string. Each character insertion
 *   into the set takes O(log K) time, where K is the number of distinct characters
 *   (a small constant, max 26 for the English alphabet). Thus, the total time is
 *   dominated by iterating through the string.
 * - Space: O(K), where K is the number of distinct characters. This is effectively
 *   constant space as the size of the character set is bounded.
 */
#include <bits/stdc++.h>

// Common competitive programming macros (mostly unused in this specific file).
#define ll long long int
#define mod 1000000007
#define pb push_back
#define mp make_pair
#define ff first
#define ss second
#define all(v) v.begin(), v.end()

using namespace std;

int main()

{
  // Speeds up C++ I/O operations, common in competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(0);

  string s;
  cin >> s;

  // A set is used to store only the unique characters from the string.
  set<char> d;
  
  /**
   * @block Distinct Character Extraction
   * @logic Iterates through the input string and inserts each character into the set.
   * The set automatically handles duplicates, ensuring only one of each character is stored.
   */
  for (auto i : s)
    d.insert(i);

  /**
   * @block Gender Determination
   * @logic Checks if the number of distinct characters is odd or even.
   * The bitwise AND `& 1` is an efficient way to check for oddness.
   */
  if (d.size() & 1)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";

  return 0;
}
