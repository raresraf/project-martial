/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A - "Chat room".
 *
 * This program checks if the word "hello" can be formed as a subsequence
 * of a given input string. It iterates through the input string and greedily
 * matches the characters of "hello" in order.
 *
 * @algorithm Greedy subsequence matching.
 * @time_complexity O(N), where N is the length of the input string, as it
 *                  involves a single pass through the string.
 * @space_complexity O(N) to store the input string.
 */
#include "bits/stdc++.h"
// PRAY :ORZ FENCING: orz orz orz orz orz
// PRAY :DOLPHINGARLIC: orz orz orz orz orz
// PRAY :DORI: orz orz orz orz orz orz orz
// PRAY :SAHIL KUCHLOUS: orz orz orz orz orz
// pray :stefan: orz orz orz orz orz orz orz
// pray :foshy: orz orz orz orz orz orz orz
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define elif else if

using namespace std;

int main() {
  // Fast I/O setup
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(0);

  int cnt{};
  const string h = "hello";
  string s;
  cin >> s;
  /**
   * @brief Greedily finds the "hello" subsequence.
   * This loop iterates through each character 'i' of the input string 's'.
   * 'cnt' tracks the current position in the target string "hello".
   * If the character 'i' matches the character at `h[cnt]`, we advance
   * our position in the target string by incrementing 'cnt'.
   */
  for (auto i : s) {
    if (i == h[cnt])
      cnt++;
  }
  /**
   * @brief Check if the full subsequence was found.
   * If 'cnt' reaches 5, it means all characters of "hello" ('h','e','l','l','o')
   * were found in the correct order.
   */
  cout << (cnt == 5 ? "YES\n" : "NO\n");
  return 0;
}
// TASKKILL /F /IM main.exe
/*
8:56 PM
9/30/2020 m/dd/yyyy
*/