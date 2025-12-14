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
  int N, a[100], i, s = 0, nb = 0, st = 0;
  cin >> N;
  for (i = 0; i < N; i++) {
    cin >> a[i];
    s += a[i];
  }
  sort(a, a + N);
  i = N - 1;
  while (i >= 0 && st <= s - st) {
    st += a[i];
    i--;
    nb++;
  }
  cout << nb << endl;
  return 0;
}
