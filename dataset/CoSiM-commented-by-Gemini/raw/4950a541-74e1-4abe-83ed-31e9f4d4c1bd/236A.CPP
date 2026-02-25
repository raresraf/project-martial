/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" programming contest problem.
 * @details The program determines a user's gender based on the number of distinct
 * characters in their username. An even number of distinct characters indicates
 * a female user ("CHAT WITH HER!"), while an odd number indicates a male user
 * ("IGNORE HIM!"). This implementation uses the C++ Standard Library's
 * sort-unique pattern to count the distinct characters.
 *
 * Algorithm:
 * 1. Read the username string.
 * 2. Copy the characters of the string into a `std::vector<char>`.
 * 3. Sort the vector. This is a prerequisite for the `std::unique` algorithm.
 * 4. Use the "erase-remove idiom":
 *    a. `std::unique` rearranges the vector so that each unique element appears
 *       once at the beginning of the range, and returns an iterator to the end
 *       of this unique range.
 *    b. `std::vector::erase` removes the now-unwanted elements from the returned
 *       iterator to the end of the vector.
 * 5. After this process, the vector's size is equal to the number of distinct characters.
 * 6. Check if the vector's size is odd or even and print the corresponding message.
 *
 * Time Complexity: O(L log L), where L is the length of the string, dominated by the sort operation.
 * Space Complexity: O(L) to store the characters in the vector.
 */
#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>
using namespace std;

/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/

int main() {
  string a;
  cin >> a;
  vector<char> vec;
  
  // Copy characters from the string to a vector.
  for (int i = 0; i < a.length(); i++) {
    vec.push_back(a.at(i));
  }
  
  // Sort the vector to prepare for the unique algorithm.
  sort(vec.begin(), vec.end());
  
  /**
   * @block
   * @description This is the "erase-remove idiom" used to find the number of
   * unique characters. `std::unique` shuffles duplicates to the end and
   * returns an iterator to the new logical end. `vec.erase` then removes them.
   */
  vec.erase(unique(vec.begin(), vec.end()), vec.end());
  
  /**
   * @block
   * @description The size of the vector now equals the count of distinct characters.
   * Check the parity of the size to determine the output.
   */
  if (vec.size() % 2 != 0) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }
  return 0;
}