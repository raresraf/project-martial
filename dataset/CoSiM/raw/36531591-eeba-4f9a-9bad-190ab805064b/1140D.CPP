#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  int n;
  cin >> n;
  int ret = 0;
  for (int i = 2; i < n; i++)
    ret += i * (i + 1);
  cout << ret << endl;
  return 0;
}
