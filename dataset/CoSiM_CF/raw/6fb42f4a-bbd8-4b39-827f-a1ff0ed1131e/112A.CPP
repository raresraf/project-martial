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
  string a, b;
  cin >> a >> b;

  for (int i = 0; i < a.size(); i++)
    a[i] = tolower(a[i]);

  for (int i = 0; i < b.size(); i++)
    b[i] = tolower(b[i]);

  int i = 0;

  while (i != a.size()) {
    if (a[i] == b[i]) {
      i++;
      continue;
    }

    if (a[i] < b[i]) {
      cout << -1;
      return 0;
    }

    if (a[i] > b[i]) {
      cout << 1;
      return 0;
    }
  }

  cout << 0;
  return 0;
}
