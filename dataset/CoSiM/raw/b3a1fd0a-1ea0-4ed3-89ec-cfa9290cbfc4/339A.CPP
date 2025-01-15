

#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); i += 2) {
    for (int j = 0; j < i; j += 2) {
      if (s[i] < s[j]) {
        swap(s[i], s[j]);
      }
    }
  }
  cout << s;
  return 0;
}