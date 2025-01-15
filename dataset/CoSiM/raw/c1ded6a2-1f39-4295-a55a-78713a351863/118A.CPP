#include <bits/stdc++.h>

using namespace std;

#define ll long long

char v[12] = {'a', 'o', 'y', 'e', 'u', 'i', 'A', 'O', 'Y', 'E', 'U', 'I'};

char down(char t) {
  if (t <= 'Z')
    return t + 32;
  else
    return t;
}

int main() {
  string s;
  cin >> s;
  bool flag;
  for (int i = 0; i < s.length(); i++) {
    flag = true;
    for (int j = 0; j < 12; j++) {
      if (s[i] == v[j]) {
        flag = false;
        break;
      }
    }
    if (flag) {
      cout << "." << down(s[i]);
    }
  }
  return 0;
}
