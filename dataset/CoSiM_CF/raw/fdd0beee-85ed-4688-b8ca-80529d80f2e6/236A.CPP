//
// Created by Conor on 5/28/2020.
//

#include <iostream>
#include <vector>

using namespace std;

int main() {
  string word;
  vector<char> chars;

  cin >> word;

  for (char &i : word) {
    bool exists = false;
    for (char j : chars) {
      if (i == j) {
        exists = true;
        break;
      }
    }

    if (!exists) {
      chars.push_back(i);
    }
  }

  if (chars.size() % 2 == 1) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }

  return 0;
}