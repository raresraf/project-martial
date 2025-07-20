#include <bits/stdc++.h>
using namespace std;
int main() {
  vector<int> v;
  int n, i, c, sum = 0, count = 0, l;
  cin >> n;
  while (n--) {
    cin >> c;
    sum += c;
    v.push_back(c);
  }
  sort(v.begin(), v.end());
  l = v.size();
  for (i = (l - 1); i >= 0; i--) {

    if (v[i] > (sum - v[i])) {
      count++;
      break;
    } else {
      v[i - 1] += v[i];
      count++;
    }
  }
  cout << count << endl;
}
