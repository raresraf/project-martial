#include <bits/stdc++.h>

using namespace std;
#define EPS (1e-7)
#define INF (1e9)
#define PI (acos(-1))

typedef long long ll;
#define endl '\n'

int dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1};

const int MOD = 1000000007;

//////////////////////////////////////////

int n, k;
int par[50];
void Input() {

  cin >> n >> k;

  for (int i = 0; i < n; i++) {

    cin >> par[i];
  }
}

void Solve() {

  int cnt = 0;

  for (int i = 0; i < n; i++) {

    if (par[i] >= par[k - 1] && par[i] > 0)
      cnt++;
  }

  cout << cnt << endl;
}

int main() {

  cin.tie(0);
  ios::sync_with_stdio(false);
  Input();
  Solve();

  return 0;
}
