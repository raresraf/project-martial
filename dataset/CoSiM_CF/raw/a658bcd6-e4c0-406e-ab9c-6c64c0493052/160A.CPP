#include <bits/stdc++.h>
using namespace std;
#define f first
#define mp make_pair
#define pb push_back
#define arlimit 1000000
typedef long long ll;
int main() {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false);
  int n;
  cin >> n;
  int ary[n];
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> ary[i];
    sum += ary[i];
  }
  int count = 0;
  int limit = sum / 2;
  int myhalf = 0;
  sort(ary, ary + n);
  bool inwhile = false;
  for (int i = n - 1; i >= 0; --i) {
    while (myhalf <= limit) {
      myhalf += ary[i];
      ++count;
      --i;
      inwhile = true;
    }
    if (inwhile) {
      ++i;
      inwhile = false;
    }
  }
  cout << count << endl;
}