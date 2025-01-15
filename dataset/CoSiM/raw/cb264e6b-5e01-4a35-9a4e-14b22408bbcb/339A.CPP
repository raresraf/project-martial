// helpful math
// 339A
// Accepted
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;
int main() {
  string s;
  cin >> s;
  vector<int> a;
  stringstream ss(s);
  int n;
  while (ss >> n) {

    // cout<<s[i];
    a.push_back(n);
    // cout<<int(s[i]-'1');
  }
  sort(a.begin(), a.end());
  int siz = a.size();
  for (int i = 0; i < siz - 1; i++) {
    cout << a[i] << "+";
  }
  cout << a[siz - 1] << endl;
}
