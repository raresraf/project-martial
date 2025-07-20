#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  char s[n + 1];
  getchar();
  scanf("%c", &s[0]);
  int cnt = 0;
  for (int i = 1; i < n; i++) {
    scanf("%c", &s[i]);
    if (s[i] == s[i - 1])
      cnt++;
  }
  cout << cnt << endl;
  return 0;
}
