#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  int n;
  cin >> n;

  while (n--) {
    string s;
    cin >> s;

    int t = s.length();

    if (t > 10)
      cout << s.at(0) << t - 2 << s.at(t - 1) << "\n";

    else
      cout << s << "\n";
  }
}
