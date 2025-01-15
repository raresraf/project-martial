#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  char s[100];
  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }
  int cnt = 0;
  for (int i = 0; i < n; i++) {
    if (s[i] == s[i + 1])
      cnt++;
  }
  cout << cnt;
}
