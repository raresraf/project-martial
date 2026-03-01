/**
 * @file 281A.CPP
 * @brief Solution for the "Word Capitalization" problem.
 *
 * @problem The task is to capitalize the first letter of a given word. If the
 * first letter is already uppercase, the word should remain unchanged.
 *
 * @algorithm The algorithm inspects the first character of the input string.
 * 1. It checks if the character is a lowercase letter by examining its ASCII value.
 *    (ASCII values for 'a'-'z' are 97-122, which are all greater than 91).
 * 2. If it is a lowercase letter, it converts it to its uppercase equivalent by
 *    subtracting 32 from its ASCII value. This works because the ASCII values for
 *    'A'-'Z' are exactly 32 less than their lowercase counterparts.
 * 3. The modified string is then printed.
 *
 * @time_complexity O(L), where L is the length of the word, dominated by I/O.
 * The capitalization logic is O(1).
 * @space_complexity O(L) to store the word string.
 */
#include <iostream>
#include <string> // Included for std::string

using namespace std;

int main() {
  string s;
  std::cin >> s;
  
  // Check if the first character s[0] is a lowercase letter.
  // In ASCII, lowercase letters ('a' = 97) have higher values than uppercase ('Z' = 90).
  // `> 91` is a simple, though not fully robust, way to check for this.
  if (int(s[0]) > 91)
    // Convert the lowercase letter to uppercase by subtracting 32 from its ASCII value.
    // A more standard way would be to use `s[0] = toupper(s[0]);`
    s[0] = int(s[0]) - 32;
    
  std::cout << s;
  return 0;
}
