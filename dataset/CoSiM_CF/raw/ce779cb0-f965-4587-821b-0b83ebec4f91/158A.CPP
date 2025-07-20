#include <iostream>
#include <vector>
using namespace std;
vector<int> v;
int k, n, e;
int score_f;
int result = 0;
int main() {
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> e;
    v.push_back(e);
  }
  for (int i = 0; i < n; i++) {
    if (v[i] > 0 && v[i] >= v[k - 1]) {
      result++;
    }
  }
  cout << result;
  return 0;
}
