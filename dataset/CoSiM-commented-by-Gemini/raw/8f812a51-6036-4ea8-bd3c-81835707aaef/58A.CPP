/**
 * @file 58A.CPP
 * @brief Solution for the "Chat room" subsequence problem (e.g., Codeforces 58A).
 * @details This file contains two attempts to solve the problem. The active solution uses a
 * clean two-pointer approach, while the commented-out "FIRST TRY" uses a flawed
 * state-machine logic. The goal is to check if "hello" is a subsequence of a given string.
 */
#include <iostream>
#include <string> // It's good practice to include the headers for the types you use.

// =================================================================================
// Active Solution: Two-Pointer Approach (Correct)
// =================================================================================
/**
 * @brief This is the primary, correct solution using a two-pointer algorithm.
 * @algorithm Two-Pointer Subsequence Check.
 * @complexity
 * - Time: O(L), where L is the length of the input string, as it's a single pass.
 * - Space: O(L) to store the input string.
 */
int main() {
  std::string a, b = "hello";
  // 'n' is the pointer/index for the target string `b`.
  // 'm' is a redundant counter for matched characters; `n` alone would suffice.
  int n = 0, m = 0;
  std::cin >> a;

  /**
   * @block Subsequence Search
   * @logic Iterates through the input string `a` with index `i`. If a character
   * matches the target character at index `n` of "hello", the target index `n` is advanced.
   */
  for (int i = 0; i < a.length(); i++) {
    if (a[i] == b[n]) {
      n++; // Move to the next character in "hello".
      m++; // Increment the count of matched characters.
    }

    // Early exit if all 5 characters have been found.
    if (m == 5) {
      break;
    }
  }

  // After the loop, if 5 characters were matched, the subsequence exists.
  if (m == 5) {
    std::cout << "YES";
  } else {
    std::cout << "NO";
  }

  return 0;
}

// =================================================================================
// Commented-out Solution: Flawed State-Machine Approach
// =================================================================================
/*
 * @brief This was the author's first attempt, which is logically flawed.
 * @details It tries to use a boolean array to track the state but fails to correctly
 * handle the requirement of finding the second 'l' after the first one. The rigid
 * `if-else if` structure does not allow for skipping characters correctly.
 *
std::string s;           // FIRST TRY
    std::cin >> s;
    bool a[5] = {false}; // Initialize array to false
    int suma = 0; // Unused variable

    for(int i = 0; i < s.length(); i++){
        if(s[i] == 'h'){
            a[0] = true;
            continue;
        }
        // This requires 'e' to come immediately after 'h', which is wrong.
        // It should just check if 'h' has been seen before (a[0] is true).
        else if(s[i] == 'e' && a[0] == true){
            a[1] = true;
            continue;
        }
        // FLAW: This logic doesn't correctly handle the two 'l's. It finds the first 'l'
        // and sets a[2], but then if the next character isn't the second 'l', it might
        // skip to the 'o' check and fail.
        else if(s[i] == 'l' && a[1] == true && a[2] == false){ // Added a check to distinguish the two 'l's
            a[2] = true;
            continue;
        }
        else if(s[i] == 'l' && a[2] == true){
            a[3] = true;
            continue;
        }
        else if(s[i] == 'o' && a[3] == true){
            a[4] = true;
            break; // Found it, can stop.
        }
    }

    if(a[4]){
        std::cout << "YES";
    }
    else{
        std::cout << "NO";
    }
*/
