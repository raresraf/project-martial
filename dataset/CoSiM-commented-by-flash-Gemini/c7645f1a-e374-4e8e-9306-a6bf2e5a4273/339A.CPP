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
  string t;
  for (int i = 0; i < s.size(); i += 2)
    t += s[i];
  sort(t.begin(), t.end());
  cout << t[0];
  for (int i = 1; i < t.size(); i++)
    cout << "+" << t[i];
  cout << endl;
  return 0;
}
