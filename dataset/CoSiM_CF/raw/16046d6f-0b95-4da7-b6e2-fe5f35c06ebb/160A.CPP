#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
vector<int> v;
void solved() {
  int n, e;
  cin >> n;
  int sum1 = 0;
  int sum2 = 0;
  for (int i = 0; i < n; i++) {
    cin >> e;
    v.push_back(e);
    sum1 += e;
  }
  int dem = 0;
  sort(v.begin(), v.end());
  for (int i = n - 1; i >= 0; i--) {
    if (sum2 <= sum1) {
      dem++;
      sum2 += v[i];
      sum1 -= v[i];
    } else {
      break;
    }
  }
  cout << dem;
}
int main() {
  ios_base::sync_with_stdio(false);
  solved();
  return 0;
}
