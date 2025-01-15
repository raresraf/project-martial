#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;
int main() {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false);
  string s, t;
  cin >> s >> t;
  for (int i = 0; i < s.size(); i++) {
    s[i] = toupper(s[i]);
    t[i] = toupper(t[i]);
  }
  for (int i = 0; i < s.size(); i++) {
    if (s[i] < t[i]) {
      cout << "-1" << endl;
      return 0;
    }
    if (s[i] > t[i]) {
      cout << "1" << endl;
      return 0;
    }
  }
  cout << 0 << endl;
}