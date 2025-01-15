#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, cnt = 0;
  string s;
  cin >> n;
  while (n--) {
    cin >> s;
    if (s[1] == '+')
      cnt++;
    else
      cnt--;
  }
  cout << cnt;
  return 0;
}
