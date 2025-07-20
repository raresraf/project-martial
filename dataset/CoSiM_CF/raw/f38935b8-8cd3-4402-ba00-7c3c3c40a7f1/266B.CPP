//
// Created by Taewoo Kim on 3/24/2017.
//

#include <iostream>

using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  while (t--) {
    int i = 0;
    while (i < n - 1) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        i++;
      }
      i++;
    }
  }
  cout << s;
  return 0;
}