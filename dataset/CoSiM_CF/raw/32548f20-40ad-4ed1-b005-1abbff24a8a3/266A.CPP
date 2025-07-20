#include <bits/stdc++.h>
using namespace std;
int main(void) {
  // your code goes here
  string s;
  int n;
  cin >> n;
  cin >> s;
  int cnt = 0;
  for (int i = 1; i < s.length(); i++) {
    if (s[i] == s[i - 1]) {
      cnt++;
      s.erase(i, 1);
      i--;
    }
  }
  cout << cnt << endl;
  return 0;
}