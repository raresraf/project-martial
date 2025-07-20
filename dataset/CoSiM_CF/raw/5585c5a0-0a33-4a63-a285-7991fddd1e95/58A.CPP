#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, h = "hello";
  getline(cin, s);
  if (s.length() < 5) {
    cout << "NO";
  } else {
    int cnt = 0;
    for (int j = 0; j < 5; j++)
      for (int i = 0; i < s.length(); i++)
        if (s[i] == h[j]) {
          j++;
          cnt++;
        }
    if (cnt == 5)
      cout << "YES";
    else
      cout << "NO";
  }
  return 0;
}
