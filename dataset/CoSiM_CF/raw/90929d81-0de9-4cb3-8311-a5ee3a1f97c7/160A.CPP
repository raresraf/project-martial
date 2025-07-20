#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

bool myfunction(int i, int j) { return (i > j); }

int main() {

  int n, sum = 0, ans = 0;
  vector<int> v;

  cin >> n;

  while (n--) {
    int a;
    cin >> a;
    v.push_back(a);
    sum += a;
  }
  sort(v.begin(), v.end(), myfunction);

  int g = 0, l = sum / 2;
  for (int i = 0; i < v.size(); i++) {
    if (g <= l) {
      g += v[i];
      ++ans;
    }
  }

  cout << ans << endl;

  return 0;
}
