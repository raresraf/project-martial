/**
 * @file 118A.CPP
 * @brief An attempt to solve the "String Task" problem.
 *
 * @details
 * This program attempts to solve the "String Task" problem, which requires
 * deleting vowels, adding a '.' before consonants, and lowercasing them.
 * The implemented logic is split into a separate function `resultString`.
 *
 * @warning The implementation is critically flawed and does not work correctly.
 * The vowel deletion logic modifies a vector while iterating over it with an
 * index, which leads to incorrect behavior and potential crashes due to
 * iterator/index invalidation. The author's own comment notes issues.
 */
#include <bits/stdc++.h>
using namespace std;

#define vi vector<int>
#define pb push_back

// not working for char 'u'

/**
 * @brief Processes a string to remove vowels and format consonants.
 * @param s The input string.
 * @return The processed (and likely incorrect) string.
 * @warning This function contains buggy logic for vowel removal.
 */
string resultString(string s) {
  vector<char> arr(s.begin(), s.end());

  /**
   * @brief First pass: Attempt to remove vowels. This block is incorrect.
   *
   * @warning The logic here is flawed. It iterates through the vector with an
   * index 'i' while simultaneously deleting elements from it. This invalidates
   * the vector's size and indices, leading to undefined behavior. The `find`
   * call also doesn't necessarily find the element at `arr[i]`, but rather
   * the *first* matching element in the vector.
   */
  for (size_t i = 0; i < arr.size(); i++) {
    switch (arr[i]) {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
    case 'y':
    case 'A':
    case 'E':
    case 'I':
    case 'O':
    case 'U':
    case 'Y':
      // This line is the source of the bug. It erases the first matching
      // character found, not necessarily the current one, and invalidates
      // the loop's state.
      arr.erase(find(arr.begin(), arr.end(), arr[i]));
      break;
    default:
      continue;
      break;
    }
  }

  /**
   * @brief Second pass: Format the remaining characters.
   *
   * Logic: This part of the function takes the (potentially corrupted) vector
   * from the first pass and builds a new string. For each remaining character,
   * it prepends a '.' and converts the character to lowercase. This logic
   * itself correctly implements the formatting part of the requirement.
   */
  vector<char> arr2;
  for (size_t i = 0; i < arr.size(); i++) {
    arr2.pb('.');
    arr2.pb(tolower(arr[i]));
  }
  string p(arr2.begin(), arr2.end());
  return p;
}

int main() {
  string s;
  cin >> s;
  cout << resultString(s) << "
";
  return 0;
}
