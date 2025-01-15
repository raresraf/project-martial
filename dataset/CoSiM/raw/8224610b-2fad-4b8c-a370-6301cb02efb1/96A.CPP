//
// Created by Taewoo Kim on 3/22/2017.
//

#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  int count = 1;
  int max = 1;
  for (int i = 1; i < s.length(); i++) {
    if (s[i] == s[i - 1]) {
      count++;
      if (count > max)
        max = count;
    } else {
      count = 1;
    }
  }
  cout << (max >= 7 ? "YES" : "NO");
  return 0;
}