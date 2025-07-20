//
// Created by Sourabh Bollapragada on 6/4/20.
//

// Problem Link https://codeforces.com/problemset/problem/282/A

#include <iostream>

using namespace std;

int main() {
  int n;
  int x = 0;
  cin >> n;
  string line;
  for (int i = 0; i < n; i++) {
    cin >> line;
    if (line == "++X" || line == "X++") {
      x++;
    } else {
      x--;
    }
  }
  cout << x << "\n";
}
