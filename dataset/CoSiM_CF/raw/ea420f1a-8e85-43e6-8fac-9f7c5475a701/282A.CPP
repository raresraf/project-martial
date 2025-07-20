#include <bits/stdc++.h>
using namespace std;
#define ll long long int
int main() {
  int n;
  cin >> n;
  int ans = 0;
  while (n--) {
    string s;
    cin >> s;
    for (int i = 0; i < s.length(); i++) {
      if (s[i] == '-') {
        ans--;
        break;
      }
      if (s[i] == '+') {
        ans++;
        break;
      }
    }
  }
  cout << ans;
}
