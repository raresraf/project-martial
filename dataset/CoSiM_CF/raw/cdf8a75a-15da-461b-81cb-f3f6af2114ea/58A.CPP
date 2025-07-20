#include <bits/stdc++.h>
using namespace std;
int main() {
  string s, str = "hello";
  int j = 0, ans = 0;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == str[j]) {
      j++;
      ans++;
      if (ans == 5)
        break;
    }
  }
  if (ans == 5)
    cout << "YES";
  else
    cout << "NO";
}
