#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef long double ld;
typedef vector<int> vi;
typedef pair<int, int> pii;
typedef pair<long long, long long> pll;
typedef vector<long long> vl;
typedef vector<pair<int, int>> vii;
#define mp make_pair
#define pb push_back
#define fr(i, j, k) for (i = j; i < k; i++)
#define FR(i, j, k) for (i = j; i <= 0; i--)
#define MOD 1000000007
#define all(v) v.begin(), v.end()

void fast() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
}

int main() {
  string temp;

  cin >> temp;

  vi a;
  for (int i = 0; i < temp.size(); i += 2) {
    a.pb(temp[i] - '0');
  }

  sort(all(a));

  for (int i = 0; i < a.size() - 1; i++) {
    cout << a[i] << '+';
  }
  cout << a[a.size() - 1];
}
