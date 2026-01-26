#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  int ans = 0;
  for (int i = 0; i <= n - 1; i++) {
    if (s[i] == s[i + 1])
      ans += 1;
  }
  cout << ans << '\n';
}