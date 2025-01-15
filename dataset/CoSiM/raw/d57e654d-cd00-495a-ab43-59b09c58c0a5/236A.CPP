
//													\[T]/
//Praise
//													 (*)
//The 													 / \  Sun

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef long double dbl;
typedef vector<int> vi;
typedef pair<int, int> pii;
typedef pair<long long, long long> pll;
typedef vector<long long> vl;
typedef vector<pair<int, int>> vii;
#define mp make_pair
#define pb push_back
#define fr(i, j, k) for (ll i = j; i < k; i++)
#define FR(i, j, k) for (ll i = j; i <= 0; i--)
#define MOD 1000000007
#define all(v) v.begin(), v.end()
#define xx first
#define yy second

void fast() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
}

int main() {
  vi vis(26, 0);

  string a;
  cin >> a;
  while (!a.empty()) {
    vis[(a[0] - 'a')] = 1;
    a.erase(0, 1);
  }

  int size;

  fr(i, 0, 26) {
    if (vis[i]) {
      size++;
    }
  }

  if (size & 1) {
    cout << "IGNORE HIM!";
    return 0;
  }

  cout << "CHAT WITH HER!";
}
