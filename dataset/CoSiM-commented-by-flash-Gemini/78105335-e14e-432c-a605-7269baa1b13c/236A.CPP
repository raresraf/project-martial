/**
 * @file 236A.CPP
 * @brief Solution to a string uniqueness problem (Codeforces problem 236A - Boy or Girl).
 * Algorithm: Counts the number of distinct characters in an input string (username).
 *            The string is sorted to bring identical characters together.
 *            Then, it iterates through the sorted string, incrementing a counter whenever
 *            a character differs from its preceding character.
 *            Finally, it determines the output based on the parity of the distinct character count.
 * Time Complexity: O(N log N) due to sorting, where N is the length of the input string.
 * Space Complexity: O(N) for storing the string (or O(1) if in-place sort is considered and string modifications are allowed).
 */
#include <bits/stdc++.h>
#include <iostream>

int main() {
  /**
   * @brief Main function to solve the "Boy or Girl" problem.
   *
   * Reads a username string, sorts its characters, counts the number of distinct
   * characters by comparing adjacent elements, and then prints
   * "IGNORE HIM!" if the count of distinct characters is even,
   * or "CHAT WITH HER!" if the count is odd.
   * @return int Returns 0 upon successful execution.
   */
  std::string username;
  std::cin >> username;
  int pass = 0;

  // Block Logic: Sorts the characters of the username string.
  // Pre-condition: 'username' contains the input string.
  // Post-condition: Characters in 'username' are arranged in non-decreasing order,
  //                 which groups identical characters together.
  std::sort(username.begin(), username.end());

  // Block Logic: Counts the number of transitions between distinct characters in the sorted string.
  // Invariant: 'pass' accumulates the count of unique characters encountered so far (minus one for the first).
  //            A commented-out alternative approach using nested loops is shown but not used.
  for (int i = 1; i < username.length(); i++) {
    if (username[i] != username[i - 1]) {
      pass++;
    }
  }

  /*
  // Commented Block: This section represents an alternative, less efficient approach
  // to count distinct characters. It iterates through each character and then
  // compares it to all preceding characters to find duplicates.
  // This is left commented as it is part of the original source, but it's
  // significantly less efficient (O(N^2)) than the sorting-based approach (O(N log N)).
  for(int i = 0; i < username.length(); i++){
      for(int j = 0; j < i; j++){
          if(username[i] == username[j]){
              n++;     //zwiekszanie wartosci tych samych slow
          }
      }

      if(n == 0){
          pass++;
      }

      n = 0;
  }
  */

  // Block Logic: Determines the output message based on the parity of 'pass'.
  // Pre-condition: 'pass' holds the number of unique character transitions.
  // Post-condition: Prints "IGNORE HIM!" if 'pass' is even, "CHAT WITH HER!" if 'pass' is odd.
  if (pass % 2 == 0) {
    std::cout << "IGNORE HIM!";
  } else {
    std::cout << "CHAT WITH HER!";
  }

  return 0;
}