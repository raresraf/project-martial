//
// Created by Conor on 5/28/2020.
//

#include <iostream>

using namespace std;

int main() {
  string word;
  cin >> word;

  string target = "hello";

  for (char c : word) {
    if (target[0] == c) {
      target = target.substr(1);
    }
  }

  if (target.length() == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}