#include <bits/stdc++.h>

using namespace std;

int initvec_and_computedis(vector<vector<int>> &vec) {
  vector<int> vectmp;
  int c, r;
  for (int i = 0; i < 5; ++i) {
    vectmp.clear();
    for (int j = 0; j < 5; ++j) {
      int tmp;
      cin >> tmp;
      if (tmp == 1) {
        r = i;
        c = j;
      }
      vectmp.push_back(tmp);
    }
    vec.push_back(vectmp);
  }
  return abs(r - 2) + abs(c - 2);
}

int main() {
  vector<vector<int>> vec;
  int ret = initvec_and_computedis(vec);
  cout << ret;
  return 0;
}