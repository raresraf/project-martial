#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  string s;
  int ans = 0;
  for (int i = 0; i < n; ++i) {
    cin >> s;
    if (s[1] == '-')
      --ans;
    else
      ++ans;
  }
  cout << ans;
}