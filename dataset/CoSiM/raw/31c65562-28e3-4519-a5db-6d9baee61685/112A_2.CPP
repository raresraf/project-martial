#include <bits/stdc++.h>
using namespace std;
int main() {
  string s, s1;
  cin >> s >> s1;
  int cnt = 0;
  for (int i = 0; i < s.size(); i++) {
    s[i] = tolower(s[i]);
    s1[i] = tolower(s1[i]);

    if (s[i] < s1[i]) {
      cout << "-1" << endl;
      break;
    } else if (s[i] > s1[i]) {
      cout << "1" << endl;
      break;
    } else
      cnt++;
  }

  if (cnt == s.size())
    cout << "0" << endl;

  return 0;
}
