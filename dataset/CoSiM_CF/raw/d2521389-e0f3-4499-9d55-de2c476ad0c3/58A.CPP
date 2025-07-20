#include <bits/stdc++.h>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  string in;
  cin >> in;
  string target = "hello";
  int idx = 0;
  for (char c : in) {
    if (target[idx] == c) {
      if (++idx >= target.size())
        break;
    }
  }
  cout << (idx >= target.size() ? "YES" : "NO");
}
