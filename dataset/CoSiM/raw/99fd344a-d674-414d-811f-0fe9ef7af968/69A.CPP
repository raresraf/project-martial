#include <bits/stdc++.h>

using namespace std;
#define py cout << "YES"
#define pn cout << "NO"
vector<int> s69a_io() {
  int times;
  cin >> times;
  int t = 3 * times;
  int ret = 0;
  vector<int> vec(3 * times + 1);
  while (t) {
    int tmp;
    cin >> tmp;
    vec[t] = tmp;
    t--;
  }
  return vec;
}
int is_xyx0(vector<int> &vec) {
  int x = 0;
  for (int i = 1; i < vec.size(); i += 3)
    x += vec[i];
  int y = 0;
  for (int i = 2; i < vec.size(); i += 3)
    y += vec[i];
  int z = 0;
  for (int i = 3; i < vec.size(); i += 3)
    z += vec[i];
  if (x == 0 && y == 0 && z == 0)
    return 0;
  else
    return -1;
}
int main() {
  vector<int> t = s69a_io();

  int val = is_xyx0(t);
  if (val)
    pn;
  else
    py;
  return 0;
}