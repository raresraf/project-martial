/**
 * @file 236A.CPP
 * @brief Determines a user's gender based on the number of distinct characters in their username.
 *
 * This program reads a username and counts the number of unique characters in it.
 * If the count of distinct characters is even, the user is classified as female
 * ("CHAT WITH HER!"). If the count is odd, the user is classified as male
 * ("IGNORE HIM!").
 *
 * @algorithm In-place duplicate character removal.
 * @complexity
 * Time: O(N^3), where N is the length of the string. The nested loops provide an
 *       O(N^2) structure, and the `string::erase` operation inside the inner
 *       loop is itself an O(N) operation, leading to a cubic time complexity.
 *       This is a highly inefficient approach compared to using a hash set/map.
 * Space: O(1) auxiliary space, as the modification happens in-place.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function to read input and process the string.
 */
int main() {
  string name;
  cin >> name;
  int length = name.length();

  // Pre-condition: 'name' holds the input string.
  // Block Logic: Use a nested loop to find and remove duplicate characters.
  for (int i = 0; i < length; i++) {
    char temp = name.at(i);
    // The inner loop checks every character after the one at index 'i'.
    for (int j = i + 1; j < length; j++) {
      char holder = name.at(j);
      // If a duplicate is found...
      if (holder == temp) {
        // ...erase it from the string. This is an O(N) operation.
        name.erase(j, 1);
        // Decrement the inner loop counter and the stored length to correctly
        // handle the shift in characters after erasure.
        j--;
        length--;
      }
    }
  }

  // Invariant: The 'name' string now contains only the unique characters
  // from the original input. Its length is the count of distinct characters.
  if (name.length() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  
  return 0;
}
