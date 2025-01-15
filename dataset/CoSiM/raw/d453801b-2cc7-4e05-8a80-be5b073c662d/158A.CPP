#include <bits/stdc++.h>
using namespace std;

int main() {
  int m, n, x, count = 0;
  vector<int> vec;
  cin >> m >> n;

  for (int i = 0; i < m; i++) {
    cin >> x;
    vec.push_back(x);
  }

  // vector<int>:: iterator itr=vec.begin();
  for (int i = 0; i < m; i++) {
    if (vec[i] >= vec[n - 1] && vec[i] > 0)
      count++;
  }

  cout << count << endl;
  return 0;
}
