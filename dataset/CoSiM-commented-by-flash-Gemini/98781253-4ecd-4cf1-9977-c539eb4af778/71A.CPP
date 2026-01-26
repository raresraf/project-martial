//
// Created by Taewoo Kim on 2/7/2017.
//

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  string s;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> s;
    if (s.size() > 10)
      s = s[0] + to_string(s.size() - 2) + s[s.size() - 1];
    cout << s << endl;
  }

  return 0;
}