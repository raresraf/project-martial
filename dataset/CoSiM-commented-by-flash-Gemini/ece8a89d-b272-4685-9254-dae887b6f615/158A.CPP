#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<string> vs;
typedef pair<int, int> pi;
typedef long long ll;
typedef unsigned long long ull;
typedef unsigned int uint;

#define sz(v) ((int)((v).size()))
#define all(v) ((v).begin(), (v).end())
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++)
#define nl << "\n"

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  int N, a[50], nb = 0, k, i, j;
  cin >> N >> k;
  for (i = 0; i < N; i++) {
    cin >> a[i];
  }
  // sort(a, a + N);
  i = 0;
  while (i < N && a[i] > 0 && i < k) {
    nb++;
    i++;
  }
  j = i--;
  while (j < N && a[j] > 0 && a[j] >= a[i]) {
    j++;
    nb++;
  }
  cout << nb << endl;
}
