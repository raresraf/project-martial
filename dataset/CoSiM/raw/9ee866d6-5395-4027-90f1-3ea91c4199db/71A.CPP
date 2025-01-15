#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
  int n;
  cin >> n;

  string s;
  for (int i = 0; i < n; ++i) {
    cin >> s;
    if (s.size() > 10) {
      cout << s[0] << s.size() - 2 << s.back();
    } else {
      cout << s;
    }

    cout << '\n';
  }
}
