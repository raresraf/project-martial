#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;

int main() {
  int t, a, b, c;
  cin >> t;
  int s1 = 0, s2 = 0, s3 = 0;
  while (t--) {
    cin >> a >> b >> c;
    s1 += a;
    s2 += b;
    s3 += c;
  }
  if (s1 == 0 && s2 == 0 && s3 == 0)
    cout << "YES\n";
  else
    cout << "NO\n";
}