// Problem: Capitalize the first letter of a given word.
// This is a competitive programming problem focusing on basic string manipulation.
// The approach is to read the input string, check if the first character is lowercase,
// and if so, convert it to uppercase.
// Time Complexity: O(1) because the length of the input string is modified at a single position.
// Space Complexity: O(N) where N is the length of the input string, to store the string.
#include <bits/stdc++.h>
using namespace std;

int main() {
  string word;

  cin >> word;

  // Check if the first character of the word is a lowercase letter.
  if (islower(word[0]))
    // If it is lowercase, convert it to its uppercase equivalent.
    word[0] = char(toupper(word[0]));

  cout << word << endl;

  return 0;
}

/**
ApPLe
ApPLe

konjac
Konjac

**/
