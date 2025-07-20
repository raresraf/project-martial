#include <bits/stdc++.h>
using namespace std;
#define F first
#define S second
#define mp make_pair
#define pb push_back
#define arsz (1 << 20)
typedef long long ll;
int main() {
  // freopen("input.txt","r", stdin);
  ios_base::sync_with_stdio(false);
  int n, t;
  string s;
  cin >> n >> t >> s;
  while (t--) {
    for (int i = 0; i < n - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        i += 1;
      }
    }
  }
  for (int i = 0; i < n; i++)
    cout << s[i];
  cout << endl;
  return 0;
}