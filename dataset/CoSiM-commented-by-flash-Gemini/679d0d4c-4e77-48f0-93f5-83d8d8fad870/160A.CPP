#include <bits/stdc++.h>
#include <vector>
using namespace std;
bool chg(int x, int y) { return x > y; }
int main() {
  vector<int> vect;
  int t, n, sum = 0, i, j;
  cin >> t;
  while (t--) {
    cin >> n;
    vect.push_back(n);
  }
  sort(vect.begin(), vect.end(), chg);
  int p = accumulate(vect.begin(), vect.end(), 0);

  for (i = 0; i < vect.size(); i++) {

    sum += vect[i];
    p = p - vect[i];
    if (sum > p) {
      cout << i + 1 << endl;
      return 0;
    }
  }

  return 0;
}
