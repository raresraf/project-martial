/**
 * @file 236A.CPP
 * @brief Determines a user's gender based on the number of distinct characters in their username.
 *
 * This program reads a username and counts the number of unique characters in it.
 * If the count of distinct characters is even, the user is classified as female
 * ("CHAT WITH HER!"). If the count is odd, the user is classified as male
 * ("IGNORE HIM!").
 *
 * @algorithm Hashing/Frequency Counting using a Map.
 * @complexity
 * Time: O(N * log K), where N is the length of the string and K is the number of
 *       distinct characters (size of the map's key set). Each map access/insertion
 *       takes logarithmic time. A std::set or std::unordered_map would also work,
 *       with std::unordered_map providing average O(N) time complexity.
 * Space: O(K), where K is the number of distinct characters, to store the map.
 *        In the worst case (all characters are unique), this is O(N).
 */
#include <iostream>
#include <string>
#include <map>

// Using the standard namespace for competitive programming convenience.
using namespace std;

/**
 * @brief Encapsulates the core logic for solving the problem.
 */
void solve() {
  // m: A map to store the frequency of each character in the string.
  map<char, int> m;
  string s;
  cin >> s;
  int n = s.length();
  // dem: A counter for the number of distinct characters.
  int dem = 0;

  // Pre-condition: 'dem' is 0, map 'm' is empty.
  // Block Logic: Iterate through the input string to count character frequencies.
  for (int i = 0; i < n; i++) {
    m[s[i]]++;
    // Invariant: 'dem' is only incremented the first time a character is seen.
    if (m[s[i]] == 1) {
      dem++;
    }
  }
  
  // Invariant: 'dem' holds the total count of unique characters in the string.
  // Block Logic: Check if the number of distinct characters is even or odd
  // to determine the output.
  if (dem % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
}

/**
 * @brief The main function, which serves as the program's entry point.
 */
int main() {
  // Calls the function that contains the problem's solution.
  solve();
  return 0;
}
