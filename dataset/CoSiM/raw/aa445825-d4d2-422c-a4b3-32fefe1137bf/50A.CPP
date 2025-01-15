#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;
int main(void) {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false);
  int n, m;
  cin >> m >> n;
  int res = floor(float(m * n) / float(2));
  cout << res << endl;
}