#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  set<char> vwl = {'a', 'e', 'i', 'o', 'u', 'y'};
  string s;
  cin >> s;
  for (auto ch : s) {
    ch = tolower(ch);
    if (vwl.find(ch) != vwl.end())
      continue;
    cout << '.' << ch;
  }
  cout << endl;
  return 0;
}
