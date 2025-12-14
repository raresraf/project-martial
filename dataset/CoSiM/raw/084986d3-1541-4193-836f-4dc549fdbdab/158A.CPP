#include <iostream>
#include <vector>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  vector<int> xs(n);
  for (int i = 0; i < n; ++i)
    cin >> xs[i];
  int val = xs[k - 1];
  int total = 0;
  for (int i = 0; i < n && xs[i] != 0 && xs[i] >= val; ++i)
    ++total;
  cout << total << endl;
  return 0;
}