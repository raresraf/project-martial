//
// Created by Conor on 5/27/2020.
//

#include <iostream>
using namespace std;

int main() {
  int n, x;
  cin >> n;
  x = 0;

  string input;

  for (int i = 0; i < n; ++i) {
    cin >> input;

    if (input.find('+') != string::npos) {
      x++;
    } else {
      x--;
    }
  }

  cout << x;

  return 0;
}