//
// Created by Conor on 5/28/2020.
//

#include <iostream>

using namespace std;

int main() {
  string word;
  cin >> word;

  if (islower(word[0])) {
    word[0] = toupper(word[0]);
  }

  cout << word;

  return 0;
}