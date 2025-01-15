#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, cnt;
  string s;
  while (cin >> n) {
    cnt = 0;
    cin >> s;
    for (int i = 0; i < n; i++)
      if (s[i] == s[i + 1])
        cnt++;

    cout << cnt << endl;
  }

  return 0;
}
