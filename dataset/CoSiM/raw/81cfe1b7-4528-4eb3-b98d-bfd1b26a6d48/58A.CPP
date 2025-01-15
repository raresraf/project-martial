#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  string s;
  cin >> s;
  string match = "hello";
  int i = 0;
  for (auto ch : s) {
    if (i >= match.size())
      break;
    if (match[i] == ch)
      i++;
  }
  cout << (i == match.size() ? "YES" : "NO") << endl;
  return 0;
}
