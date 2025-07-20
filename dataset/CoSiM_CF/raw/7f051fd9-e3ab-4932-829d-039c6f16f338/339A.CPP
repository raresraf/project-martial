#include <bits/stdc++.h>
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
#define mod 1000000007
using namespace std;
int main() {
  std::ios_base::sync_with_stdio(false); // cin.tie(NULL);
  int n, i, j = 0;
  string s;
  cin >> s;
  int len = s.length();
  int alen = (len + 1) / 2;
  int a[alen];
  loop(alen) {
    a[i] = s[j] - '0';
    j += 2;
  }
  sort(a, a + alen);
  loop(alen) {
    cout << a[i];
    if (i != alen - 1)
      cout << "+";
  }
  cout << "\n";
  return 0;
}