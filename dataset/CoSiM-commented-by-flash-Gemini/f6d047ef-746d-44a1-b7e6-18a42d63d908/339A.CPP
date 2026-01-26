#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  int cnt = 0;
  vector<int> vec;
  getline(cin, s);
  stringstream ss(s);
  string ind;
  while (getline(ss, ind, '+')) {
    int a = stoi(ind);
    vec.push_back(a);
  }
  sort(vec.begin(), vec.end());
  cout << vec[0];
  for (int i = 1; i < vec.size(); i++) {
    cout << "+" << vec[i];
  }
  cout << endl;

  return 0;
}
