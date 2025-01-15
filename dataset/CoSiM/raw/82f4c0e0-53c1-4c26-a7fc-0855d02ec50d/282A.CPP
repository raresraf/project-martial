//
// Created by Taewoo Kim on 3/21/2017.
//

#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int t = 0;
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    t += s[1] == '+' ? 1 : -1;
  }
  cout << t;

  return 0;
}