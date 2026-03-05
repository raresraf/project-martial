/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem by counting distinct characters in a username.
 *
 * This program determines a user's gender based on the number of distinct characters
 * in their username. If the count is even, the user is considered female ("CHAT WITH HER!").
 * If the count is odd, the user is considered male ("IGNORE HIM!").
 *
 * @algorithm
 * This implementation uses sorting to efficiently count the distinct characters.
 *
 * 1. Read the input username string.
 * 2. Sort the string alphabetically. This operation groups all identical characters
 *    together (e.g., "hello" becomes "ehllo").
 * 3. Initialize a counter `pass` to 0. This counter will track the number of
 *    transitions between different characters.
 * 4. Iterate through the sorted string from the second character (index 1) to the end.
 * 5. Compare each character `username[i]` with the one before it, `username[i-1]`.
 * 6. If the characters are different, it marks the end of a block of identical
 *    characters, so we increment `pass`.
 * 7. After the loop, `pass + 1` gives the total number of distinct characters.
 *    For "ehllo", the transitions are e->h, h->l, l->o. So, `pass` will be 3.
 *    The number of distinct characters is `pass + 1` = 4.
 * 8. Finally, check the parity of the number of distinct characters.
 *    - The code checks `if (pass % 2 == 0)`.
 *    - If `pass` is even, `pass + 1` is odd. This is the male case ("IGNORE HIM!").
 *    - If `pass` is odd, `pass + 1` is even. This is the female case ("CHAT WITH HER!").
 *    The logic correctly maps the parity of `pass` to the required output.
 *
 * @note
 * The file contains a commented-out block showing an alternative O(N^2) algorithm
 * for counting distinct characters, which is much less efficient than the O(N log N)
 * sorting approach used.
 *
 * @complexity
 * Time Complexity: O(L log L), where L is the length of the username, dominated by `std::sort`.
 * Space Complexity: O(L) to store the username string.
 */

#include <bits/stdc++.h>
#include <iostream>

int main() {
  std::string username;
  std::cin >> username;
  
  // `pass` will count the number of changes between characters in the sorted string.
  // The number of distinct characters will be `pass + 1`.
  int pass = 0;

  // Sort the string to group identical characters.
  std::sort(username.begin(), username.end());

  /**
   * @brief Iterate through the sorted string to count character changes.
   */
  for (int i = 1; i < username.length(); i++) {
    if (username[i] != username[i - 1]) {
      pass++;
    }
  }
  
  /* --- Commented out O(N^2) alternative from original code ---
  for(int i = 0; i < username.length(); i++){
      for(int j = 0; j < i; j++){
          if(username[i] == username[j]){
              n++;     //zwiekszanie wartosci tych samych slow (Polish: "increasing value of same words")
          }
      }

      if(n == 0){
          pass++;
      }

      n = 0;
  }*/
  
  // Let N be the number of distinct characters, so N = pass + 1.
  // If N is odd, we ignore him. If N is even, we chat with her.
  // This is equivalent to checking the parity of `pass`.
  if (pass % 2 == 0) {
    // `pass` is even => N = pass + 1 is odd => Male
    std::cout << "IGNORE HIM!";
  } else {
    // `pass` is odd => N = pass + 1 is even => Female
    std::cout << "CHAT WITH HER!";
  }

  return 0;
}
