/**
 * @file 236A.CPP
 * @brief Solution to a competitive programming problem based on the number of distinct characters in a username.
 * @details The program determines whether to "chat with her" or "ignore him" based on the number of
 * distinct characters in a given username. If the number of distinct characters is even, the decision is
 * to chat. If it's odd, the decision is to ignore.
 *
 * @algorithm Count distinct characters in a string.
 * @complexity
 * Time: O(L log L), where L is the length of the string, dominated by the sorting step.
 * Space: O(L) to store the string, or O(1) if modifying the input string in-place is considered.
 */
#include <iostream>
#include <string>
#include <algorithm> // For std::sort and std::unique
#include <vector>    // Often used with algorithms, good to be aware of

using namespace std;

int main() {
  string s;
  cin >> s;

  // Sort the string to bring all identical characters together.
  // For example, "banana" becomes "aaabnn".
  sort(s.begin(), s.end());

  // std::unique reorders the string so that each unique character appears once at the
  // beginning of the string, and returns an iterator to the end of the unique range.
  // s.erase then removes the duplicate characters from the end of the string.
  // After this, 's' contains only the unique characters from the original input.
  s.erase(unique(s.begin(), s.end()), s.end());

  /**
   * @brief Check if the number of distinct characters is even or odd.
   * The length of the modified string 's' now represents the count of distinct characters.
   */
  if (s.length() % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }

  return 0; // Indicate successful execution.
}