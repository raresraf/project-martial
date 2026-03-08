/**
 * @4950a541-74e1-4abe-83ed-31e9f4d4c1bd/236A.CPP
 * @brief This program determines an output message based on the parity of
 *        the number of unique characters in an input string.
 * Algorithm: Extracts unique characters using sorting and `std::unique`, then checks the count's parity.
 * Time Complexity: O(L log L) where L is the length of the input string due to sorting.
 * Space Complexity: O(L) for storing characters in a vector.
 */
#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  string a; // Declare a string variable 'a' to store the input string.
  // Functional Utility: Read the input string.
  cin >> a;
  vector<char> vec; // Declare a vector of characters to store individual characters of the string.
  // Block Logic: Populate the vector 'vec' with characters from the input string 'a'.
  for (int i = 0; i < a.length(); i++) {
    vec.push_back(a.at(i));
  }
  // Block Logic: Sort the vector to group identical characters together.
  // This is a prerequisite for `unique` to work correctly.
  sort(vec.begin(), vec.end());
  // Block Logic: Remove duplicate characters from the sorted vector.
  // `unique` moves all unique elements to the beginning and returns an iterator to the end of the unique range.
  // `erase` then removes the redundant elements from the vector.
  vec.erase(unique(vec.begin(), vec.end()), vec.end());
  // Block Logic: Check the parity of the count of unique characters to determine the output message.
  if (vec.size() % 2 != 0) { // If the number of unique characters is odd.
    cout << "IGNORE HIM!";
  } else { // If the number of unique characters is even.
    cout << "CHAT WITH HER!";
  }
  return 0; // Indicate successful program termination.
}
